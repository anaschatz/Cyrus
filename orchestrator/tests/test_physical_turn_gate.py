from __future__ import annotations

import json
import sqlite3
import tempfile
import threading
import time
import unittest
import uuid
from contextlib import closing
from pathlib import Path
from unittest import mock

import _bootstrap  # noqa: F401
from helpers import payload

from jarvis_orchestrator import store as store_module
from jarvis_orchestrator.config import OrchestratorConfig
from jarvis_orchestrator.domain import (
    InvalidStateError,
    OrchestratorError,
    TaskState,
    utc_now,
)
from jarvis_orchestrator.physical_turn_gate import (
    VoiceConnectionState,
    _live_voice_state,
    _open_read_only,
    run_macos_physical_turn_gate,
)
from jarvis_orchestrator.store import OrchestratorStore

TRACKER_ONE = "00000000-0000-4000-8000-000000000001"
TRACKER_TWO = "00000000-0000-4000-8000-000000000002"


class FakeStack:
    def __init__(self, config: Path) -> None:
        self.orchestrator_config_path = config

    @staticmethod
    def deployment_identity() -> dict[str, str]:
        return {"deployment_sha256": "d" * 64}


class FakeHttpResponse:
    def __init__(self, body: bytes, *, status: int = 200) -> None:
        self.body = body
        self.status = status

    def __enter__(self) -> "FakeHttpResponse":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self, _maximum: int) -> bytes:
        return self.body


class PhysicalTurnGateTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.database = self.root / "state.sqlite3"
        self.store = OrchestratorStore(self.database)
        self.addCleanup(self.store.close)
        self.config = self.root / "orchestrator.json"
        self.config.write_text(
            json.dumps(
                {
                    "database_path": str(self.database),
                    "control_bearer_token": "AAECAwQFBgcICQoLDA0ODxAREhMUFRYXGBkaGxwdHh8",
                    "inference_bearer_token": "ICEiIyQlJicoKSorLC0uLzAxMjM0NTY3ODk6Ozw9Pj8",
                }
            ),
            encoding="utf-8",
        )
        self.config.chmod(0o600)
        self.stack = FakeStack(self.config)

    @staticmethod
    def _connected_voice_state() -> VoiceConnectionState:
        return VoiceConnectionState(
            connected=True,
            tracker_id=TRACKER_ONE,
            generation=1,
        )

    def _create_genuine_v7_database(self) -> Path:
        database = self.root / "legacy-v7.sqlite3"
        connection = sqlite3.connect(database, isolation_level=None)
        try:
            connection.execute("PRAGMA foreign_keys=ON")
            connection.execute("BEGIN IMMEDIATE")
            store_module._execute_script_tx(connection, store_module.SCHEMA_SQL)
            connection.execute(
                "INSERT INTO schema_meta(key,value) VALUES('schema_version','1')"
            )
            store_module._execute_script_tx(connection, store_module.MIGRATION_2_SQL)
            OrchestratorStore._set_schema_version_tx(connection, 1, 2)
            store_module._execute_script_tx(connection, store_module.MIGRATION_3_SQL)
            OrchestratorStore._set_schema_version_tx(connection, 2, 3)
            connection.execute(
                "ALTER TABLE tasks ADD COLUMN cloud_dispatch_count "
                "INTEGER NOT NULL DEFAULT 0 CHECK("
                "cloud_dispatch_count >= 0 AND cloud_dispatch_count <= 1)"
            )
            store_module._execute_script_tx(connection, store_module.MIGRATION_4_SQL)
            OrchestratorStore._set_schema_version_tx(connection, 3, 4)
            store_module._execute_script_tx(connection, store_module.MIGRATION_5_SQL)
            OrchestratorStore._set_schema_version_tx(connection, 4, 5)
            store_module._execute_script_tx(connection, store_module.MIGRATION_6_SQL)
            OrchestratorStore._set_schema_version_tx(connection, 5, 6)
            store_module._execute_script_tx(connection, store_module.MIGRATION_7_SQL)
            OrchestratorStore._set_schema_version_tx(connection, 6, 7)
            connection.execute("COMMIT")
        except BaseException:
            connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()
        database.chmod(0o600)
        return database

    def _complete_physical_turn(
        self, *, secret: str = "SECRET TRANSCRIPT"
    ) -> tuple[str, str, str, str]:
        input_event_id = str(uuid.uuid4())
        turn_id = str(uuid.uuid4())
        admitted = self.store.admit_transcript(
            event_id=input_event_id,
            turn_id=turn_id,
            payload=payload(secret),
            now=utc_now(),
        ).task
        assert admitted is not None
        task = self.store.transition(
            admitted.task_id,
            admitted.epoch,
            TaskState.ADMITTED,
            TaskState.ROUTING,
        )
        task = self.store.transition(
            task.task_id,
            task.epoch,
            TaskState.ROUTING,
            TaskState.LOADING_CONTEXT,
        )
        task = self.store.transition(
            task.task_id,
            task.epoch,
            TaskState.LOADING_CONTEXT,
            TaskState.EXECUTING,
        )
        reservation = self.store.reserve_model(
            task.task_id, task.epoch, purpose="final", output_allowance=32
        )
        self.store.mark_model_dispatched(reservation, task.task_id, task.epoch)
        self.store.reconcile_model(reservation, task.task_id, task.epoch, 4)
        self.store.start_response(task.task_id, task.epoch, "el-GR")
        self.store.append_response_event(
            task.task_id,
            task.epoch,
            "response.text.delta",
            {"chunk_index": 0, "text": secret},
        )
        self.store.append_response_event(
            task.task_id,
            task.epoch,
            "response.speech.commit",
            {"unit_index": 1, "locale": "el-GR", "text": secret},
        )
        self.store.complete_response(
            task.task_id,
            task.epoch,
            secret,
            chunk_count=1,
            trailing_speech=[],
            last_unit_index=1,
        )
        claimed = self.store.claim_outbox(task.conversation_id, "native-voice")
        for event in claimed:
            self.store.finish_outbox_attempt(
                event.event_id, "native-voice", complete=True
            )
        playback_event_id = str(uuid.uuid4())
        self.store.record_voice_playback_terminal(
            event_id=playback_event_id,
            semantic_sha256="a" * 64,
            response_id=task.response_id,
            terminal_type="playback.completed",
            client_sent_at=utc_now().isoformat().replace("+00:00", "Z"),
        )
        return input_event_id, turn_id, task.task_id, task.response_id

    def test_one_new_completed_turn_is_proved_without_content(self) -> None:
        identities: list[str] = []

        def complete() -> None:
            time.sleep(0.03)
            identities.extend(self._complete_physical_turn())

        worker = threading.Thread(target=complete, daemon=True)
        worker.start()
        report = run_macos_physical_turn_gate(
            self.stack,  # type: ignore[arg-type]
            timeout_seconds=2,
            poll_interval_seconds=0.005,
            settle_seconds=0.01,
            voice_state=self._connected_voice_state,
        )
        worker.join(timeout=1)
        self.assertTrue(report["ready"], report)
        self.assertEqual(report["version"], "jarvis-macos-physical-turn/2")
        self.assertTrue(all(report["evidence"].values()))
        self.assertNotIn("SECRET TRANSCRIPT", json.dumps(report))
        for identifier in identities:
            self.assertNotIn(identifier, json.dumps(report))
        self.assertEqual(report["deployment_sha256"], "d" * 64)
        self.assertEqual(report["event_counts"]["response.speech.commit"], 1)

    def test_disconnect_reconnect_between_polls_fails_closed(self) -> None:
        probe_calls = 0

        def reconnected_state() -> VoiceConnectionState:
            nonlocal probe_calls
            probe_calls += 1
            return VoiceConnectionState(
                connected=True,
                tracker_id=TRACKER_ONE,
                generation=1 if probe_calls <= 2 else 3,
            )

        worker = threading.Thread(
            target=lambda: (time.sleep(0.03), self._complete_physical_turn()),
            daemon=True,
        )
        worker.start()
        report = run_macos_physical_turn_gate(
            self.stack,  # type: ignore[arg-type]
            timeout_seconds=2,
            poll_interval_seconds=0.005,
            settle_seconds=0.01,
            voice_state=reconnected_state,
        )
        worker.join(timeout=1)

        self.assertFalse(report["ready"])
        self.assertTrue(report["evidence"]["voice_connected_at_start"])
        self.assertTrue(report["evidence"]["voice_connected_at_end"])
        self.assertFalse(
            report["evidence"]["voice_connection_continuous_during_gate"]
        )
        self.assertIn(
            "evidence_failed.voice_connection_continuous_during_gate",
            report["blockers"],
        )
        encoded = json.dumps(report)
        self.assertNotIn("voice_connection_generation", encoded)
        self.assertNotIn("voice_connection_tracker_id", encoded)
        self.assertNotIn(TRACKER_ONE, encoded)

    def test_connection_tracker_restart_during_gate_fails_closed(self) -> None:
        probe_calls = 0

        def restarted_state() -> VoiceConnectionState:
            nonlocal probe_calls
            probe_calls += 1
            return VoiceConnectionState(
                connected=True,
                tracker_id=TRACKER_ONE if probe_calls <= 2 else TRACKER_TWO,
                generation=1,
            )

        worker = threading.Thread(
            target=lambda: (time.sleep(0.03), self._complete_physical_turn()),
            daemon=True,
        )
        worker.start()
        report = run_macos_physical_turn_gate(
            self.stack,  # type: ignore[arg-type]
            timeout_seconds=2,
            poll_interval_seconds=0.005,
            settle_seconds=0.01,
            voice_state=restarted_state,
        )
        worker.join(timeout=1)

        self.assertFalse(report["ready"])
        self.assertFalse(
            report["evidence"]["voice_connection_continuous_during_gate"]
        )
        self.assertIn(
            "evidence_failed.voice_connection_continuous_during_gate",
            report["blockers"],
        )

    def test_live_voice_state_parser_is_exact_and_fail_closed(self) -> None:
        config = OrchestratorConfig.load(self.config)
        valid = json.dumps(
            {
                "core_ready": True,
                "effective_capabilities": {
                    "cloud": False,
                    "memory_retrospective": False,
                    "memory_shadow": False,
                    "research": False,
                    "tools": False,
                },
                "schema_version": "jarvis-orchestrator-beta-session/2",
                "voice_client_connected": True,
                "voice_connection_generation": 7,
                "voice_connection_tracker_id": TRACKER_ONE,
            }
        ).encode("utf-8")
        with mock.patch(
            "jarvis_orchestrator.physical_turn_gate.urllib.request.urlopen",
            return_value=FakeHttpResponse(valid),
        ) as opener:
            state = _live_voice_state(config)
        self.assertEqual(state, VoiceConnectionState(True, TRACKER_ONE, 7))
        self.assertTrue(opener.call_args.args[0].full_url.endswith("/v1/beta-session"))

        not_ready = json.loads(valid)
        not_ready["core_ready"] = False
        invalid_tracker = json.loads(valid)
        invalid_tracker["voice_connection_tracker_id"] = "not-a-uuid"

        for response in (
            FakeHttpResponse(valid, status=503),
            FakeHttpResponse(b"x" * 4097),
            FakeHttpResponse(b'{}'),
            FakeHttpResponse(json.dumps(not_ready).encode("utf-8")),
            FakeHttpResponse(json.dumps(invalid_tracker).encode("utf-8")),
        ):
            with self.subTest(status=response.status, size=len(response.body)):
                with mock.patch(
                    "jarvis_orchestrator.physical_turn_gate.urllib.request.urlopen",
                    return_value=response,
                ):
                    self.assertIsNone(_live_voice_state(config))

    def test_timeout_is_content_free_and_fail_closed(self) -> None:
        report = run_macos_physical_turn_gate(
            self.stack,  # type: ignore[arg-type]
            timeout_seconds=0.03,
            poll_interval_seconds=0.005,
            settle_seconds=0,
            voice_state=self._connected_voice_state,
        )
        self.assertFalse(report["ready"])
        self.assertIn("physical_turn_timeout", report["blockers"])
        self.assertFalse(report["evidence"]["exactly_one_new_turn"])

    def test_read_only_database_paths_are_encoded_without_creating_files(self) -> None:
        for name in ("state#review.sqlite3", "state?review.sqlite3", "state%23review.sqlite3", "βάση.sqlite3"):
            with self.subTest(name=name):
                path = self.root / name
                with closing(sqlite3.connect(path, isolation_level=None)) as connection:
                    connection.execute("CREATE TABLE marker(value)")
                    connection.execute("INSERT INTO marker VALUES(?)", (name,))
                path.chmod(0o600)
                before = set(self.root.iterdir())
                with closing(_open_read_only(path)) as connection:
                    self.assertEqual(connection.execute("SELECT value FROM marker").fetchone()[0], name)
                    with self.assertRaises(sqlite3.OperationalError):
                        connection.execute("CREATE TABLE forbidden(value)")
                self.assertEqual(set(self.root.iterdir()), before)

    def test_more_than_one_new_admission_fails_closed(self) -> None:
        def admit_twice() -> None:
            time.sleep(0.02)
            for index in range(2):
                self.store.admit_transcript(
                    event_id=str(uuid.uuid4()),
                    turn_id=str(uuid.uuid4()),
                    payload=payload(f"private-{index}"),
                    now=utc_now(),
                )

        worker = threading.Thread(target=admit_twice, daemon=True)
        worker.start()
        report = run_macos_physical_turn_gate(
            self.stack,  # type: ignore[arg-type]
            timeout_seconds=1,
            poll_interval_seconds=0.01,
            settle_seconds=0,
            voice_state=self._connected_voice_state,
        )
        worker.join(timeout=1)
        self.assertFalse(report["ready"])
        self.assertIn("multiple_new_voice_turns", report["blockers"])
        self.assertFalse(report["evidence"]["exactly_one_new_turn"])

    def test_playback_receipt_is_idempotent_and_content_free(self) -> None:
        self._complete_physical_turn(secret="PRIVATE RESPONSE")
        with self.store._lock:
            row = self.store._connection.execute(
                """SELECT event_id,semantic_sha256,task_id,epoch,response_id,
                          terminal_type,client_sent_at,received_at
                   FROM voice_playback_receipts"""
            ).fetchone()
            columns = {
                item[1]
                for item in self.store._connection.execute(
                    "PRAGMA table_info(voice_playback_receipts)"
                ).fetchall()
            }
            task_event = self.store._connection.execute(
                """SELECT event_type,payload_json FROM task_events
                   WHERE event_type='voice.playback.completed'"""
            ).fetchone()
        self.assertIsNotNone(row)
        self.assertNotIn("payload_json", columns)
        self.assertEqual(task_event["payload_json"], "{}")
        self.assertNotIn("PRIVATE RESPONSE", repr(tuple(row)))
        self.assertEqual(
            self.store.record_voice_playback_terminal(
                event_id=row["event_id"],
                semantic_sha256=row["semantic_sha256"],
                response_id=row["response_id"],
                terminal_type=row["terminal_type"],
                client_sent_at=utc_now().isoformat().replace("+00:00", "Z"),
            ),
            "duplicate",
        )
        with self.assertRaisesRegex(
            OrchestratorError, "protocol.conflicting_duplicate"
        ):
            self.store.record_voice_playback_terminal(
                event_id=row["event_id"],
                semantic_sha256="f" * 64,
                response_id=row["response_id"],
                terminal_type=row["terminal_type"],
                client_sent_at=utc_now().isoformat().replace("+00:00", "Z"),
            )
        with self.assertRaisesRegex(
            OrchestratorError, "protocol.conflicting_duplicate"
        ):
            self.store.record_voice_playback_terminal(
                event_id=str(uuid.uuid4()),
                semantic_sha256="b" * 64,
                response_id=row["response_id"],
                terminal_type="playback.interrupted",
                client_sent_at=utc_now().isoformat().replace("+00:00", "Z"),
            )

    def test_completed_requires_completed_task_and_speech_done(self) -> None:
        admitted = self.store.admit_transcript(
            event_id=str(uuid.uuid4()),
            turn_id=str(uuid.uuid4()),
            payload=payload(),
        ).task
        assert admitted is not None
        task = self.store.transition(
            admitted.task_id,
            admitted.epoch,
            TaskState.ADMITTED,
            TaskState.ROUTING,
        )
        task = self.store.transition(
            task.task_id,
            task.epoch,
            TaskState.ROUTING,
            TaskState.LOADING_CONTEXT,
        )
        task = self.store.transition(
            task.task_id,
            task.epoch,
            TaskState.LOADING_CONTEXT,
            TaskState.EXECUTING,
        )
        self.store.start_response(task.task_id, task.epoch, "el-GR")
        with self.assertRaises(InvalidStateError):
            self.store.record_voice_playback_terminal(
                event_id=str(uuid.uuid4()),
                semantic_sha256="c" * 64,
                response_id=task.response_id,
                terminal_type="playback.completed",
                client_sent_at=utc_now().isoformat().replace("+00:00", "Z"),
            )

    def test_interrupted_receipt_remains_distinct_from_completed(self) -> None:
        admitted = self.store.admit_transcript(
            event_id=str(uuid.uuid4()),
            turn_id=str(uuid.uuid4()),
            payload=payload(),
        ).task
        assert admitted is not None
        status = self.store.record_voice_playback_terminal(
            event_id=str(uuid.uuid4()),
            semantic_sha256="e" * 64,
            response_id=admitted.response_id,
            terminal_type="playback.interrupted",
            client_sent_at=utc_now().isoformat().replace("+00:00", "Z"),
        )
        self.assertEqual(status, "accepted")
        with self.store._lock:
            row = self.store._connection.execute(
                """SELECT terminal_type FROM voice_playback_receipts
                   WHERE response_id=?""",
                (admitted.response_id,),
            ).fetchone()
        self.assertEqual(row["terminal_type"], "playback.interrupted")

    def test_current_v10_keeps_content_free_playback_receipts(self) -> None:
        with self.store._lock:
            version = self.store._connection.execute(
                "SELECT value FROM schema_meta WHERE key='schema_version'"
            ).fetchone()[0]
            columns = {
                row[1]
                for row in self.store._connection.execute(
                    "PRAGMA table_info(voice_playback_receipts)"
                ).fetchall()
            }
        self.assertEqual(version, "10")
        self.assertNotIn("payload_json", columns)

    def test_genuine_v7_database_migrates_to_current_content_free_receipts(
        self,
    ) -> None:
        database = self._create_genuine_v7_database()
        with sqlite3.connect(database) as legacy:
            version = legacy.execute(
                "SELECT value FROM schema_meta WHERE key='schema_version'"
            ).fetchone()[0]
            memory_columns = {
                row[1]
                for row in legacy.execute("PRAGMA table_info(memory_records)").fetchall()
            }
            receipt_table = legacy.execute(
                "SELECT 1 FROM sqlite_master "
                "WHERE type='table' AND name='voice_playback_receipts'"
            ).fetchone()
        self.assertEqual(version, "7")
        self.assertIn("scoped_user_id", memory_columns)
        self.assertIn("metadata_json", memory_columns)
        self.assertNotIn("principal_id", memory_columns)
        self.assertIsNone(receipt_table)

        migrated = OrchestratorStore(database)
        self.addCleanup(migrated.close)
        with migrated._lock:
            version = migrated._connection.execute(
                "SELECT value FROM schema_meta WHERE key='schema_version'"
            ).fetchone()[0]
            receipt_columns = {
                row[1]
                for row in migrated._connection.execute(
                    "PRAGMA table_info(voice_playback_receipts)"
                ).fetchall()
            }

        self.assertEqual(version, str(store_module.SCHEMA_VERSION))
        self.assertIn("terminal_type", receipt_columns)
        self.assertNotIn("payload_json", receipt_columns)


if __name__ == "__main__":
    unittest.main()
