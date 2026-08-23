from __future__ import annotations

import json
import plistlib
import tempfile
import unittest
import uuid
from pathlib import Path

from _bootstrap import ROOT  # noqa: F401

from jarvis_orchestrator.background import BackgroundServiceManager
from jarvis_orchestrator.domain import TaskState
from jarvis_orchestrator.operations import (
    ContentFreeOperationalLog,
    OperationalError,
    ProcessLock,
    backup_database,
    database_diagnostics,
    restore_database,
)
from jarvis_orchestrator.readiness import (
    run_live_sequential_gate,
    run_sqlite_recovery_gate,
    run_synthetic_lifecycle_gate,
)
from jarvis_orchestrator.store import OrchestratorStore
from jarvis_voice.voice_auth import VoiceAuthKey


def _completed_store(path: Path) -> OrchestratorStore:
    store = OrchestratorStore(path)
    task = store.admit_transcript(
        event_id=str(uuid.uuid4()),
        turn_id=str(uuid.uuid4()),
        payload={
            "asr_score": {
                "calibrated": False,
                "method": "test",
                "value": 1.0,
            },
            "language": {"code": "en", "score": 1.0},
            "text": "operational test",
        },
    ).task
    assert task is not None
    task = store.transition(task.task_id, task.epoch, TaskState.ADMITTED, TaskState.ROUTING)
    task = store.transition(
        task.task_id, task.epoch, TaskState.ROUTING, TaskState.LOADING_CONTEXT
    )
    task = store.transition(
        task.task_id, task.epoch, TaskState.LOADING_CONTEXT, TaskState.EXECUTING
    )
    store.start_response(task.task_id, task.epoch, "en-US")
    store.append_response_event(
        task.task_id,
        task.epoch,
        "response.text.delta",
        {"chunk_index": 0, "text": "Ready."},
    )
    store.complete_response(
        task.task_id,
        task.epoch,
        "Ready.",
        chunk_count=1,
        trailing_speech=[(1, "en-US", "Ready.")],
        last_unit_index=1,
    )
    return store


class OperationalPersistenceTests(unittest.TestCase):
    def test_diagnostics_are_content_free_and_detect_clean_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.sqlite3"
            store = _completed_store(path)
            store.checkpoint()
            report = database_diagnostics(path)
            store.close()
            self.assertTrue(report.ready)
            encoded = json.dumps(report.to_dict(), sort_keys=True)
            self.assertNotIn("operational test", encoded)
            self.assertNotIn("Ready.", encoded)
            self.assertEqual(report.invariant_violation_count, 0)

    def test_diagnostics_fail_closed_on_memory_crypto_state_corruption(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.sqlite3"
            store = _completed_store(path)
            with store._transaction() as connection:
                connection.execute("DELETE FROM memory_crypto_state")
            store.checkpoint()

            report = database_diagnostics(path)
            store.close()

            self.assertFalse(report.ready)
            self.assertEqual(
                report.invariant_counts["memory_crypto_state_cardinality"],
                1,
            )
            self.assertGreater(report.invariant_violation_count, 0)

    def test_online_backup_and_offline_restore_are_verified(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database = root / "state.sqlite3"
            backup = root / "backup.sqlite3"
            lock = root / "service.lock"
            store = _completed_store(database)
            backup_report = backup_database(database, backup)
            self.assertTrue(backup_report["integrity_ok"])
            store.close()
            database.write_bytes(b"not-a-sqlite-database")
            with self.assertRaises(OperationalError):
                database_diagnostics(database)
            restore_report = restore_database(backup, database, service_lock=lock)
            self.assertTrue(restore_report["integrity_ok"])
            self.assertTrue(database_diagnostics(database).ready)

    def test_recovery_gate_corrupts_only_a_disposable_copy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "state.sqlite3"
            store = _completed_store(database)
            report = run_sqlite_recovery_gate(database)
            self.assertTrue(report["ready"])
            self.assertTrue(report["corruption_detected"])
            self.assertFalse(report["production_database_mutated"])
            self.assertTrue(database_diagnostics(database).ready)
            store.close()

    def test_restore_fails_while_service_lock_is_held(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database = root / "state.sqlite3"
            backup = root / "backup.sqlite3"
            lock_path = root / "service.lock"
            store = _completed_store(database)
            backup_database(database, backup)
            store.close()
            with ProcessLock(lock_path), self.assertRaisesRegex(
                OperationalError, "service_already_running"
            ):
                restore_database(backup, database, service_lock=lock_path)

    def test_content_free_log_rejects_strings_and_rotates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "operations.jsonl"
            logger = ContentFreeOperationalLog(path, maximum_bytes=256, backup_count=2)
            with self.assertRaises(ValueError):
                logger.emit("bad", prompt="private")  # type: ignore[arg-type]
            for index in range(100):
                logger.emit("queue_sample", depth=index)
            logger.close()
            self.assertLessEqual(len(list(path.parent.glob("operations.jsonl*"))), 3)
            self.assertNotIn("private", "".join(item.read_text() for item in path.parent.glob("operations.jsonl*")))

    def test_background_runner_contains_paths_but_no_configuration_secrets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "config.json"
            config.write_text('{"control_bearer_token":"must-not-be-copied"}', encoding="utf-8")
            manager = BackgroundServiceManager(
                config_path=config,
                database_path=root / "state.sqlite3",
            )
            manager._write_runner()
            runner = manager.runner.read_text(encoding="utf-8")
            self.assertIn(str(config), runner)
            self.assertNotIn("must-not-be-copied", runner)
            self.assertIn("jarvis_orchestrator.cli", runner)

    def test_background_specs_have_controlled_restart_and_bounded_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "config.json"
            config.write_text("{}", encoding="utf-8")

            class TestManager(BackgroundServiceManager):
                @property
                def macos_plist(self) -> Path:
                    return root / "com.jarvis.orchestrator.plist"

            manager = TestManager(
                config_path=config,
                database_path=root / "state.sqlite3",
            )
            manager._write_runner()
            manager._write_macos_plist()
            with manager.macos_plist.open("rb") as source:
                plist = plistlib.load(source)
            self.assertEqual(plist["KeepAlive"], {"SuccessfulExit": False})
            self.assertTrue(plist["RunAtLoad"])
            self.assertEqual(plist["StandardOutPath"], "/dev/null")
            manager._write_windows_xml()
            task_xml = manager.windows_xml.read_text(encoding="utf-16")
            self.assertIn("<MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>", task_xml)
            self.assertIn("<RestartOnFailure>", task_xml)
            self.assertIn("<ExecutionTimeLimit>PT0S</ExecutionTimeLimit>", task_xml)


class SyntheticReadinessTests(unittest.TestCase):
    def test_all_durable_cancellation_stages_are_exercised(self) -> None:
        report = run_synthetic_lifecycle_gate(turns=90)
        self.assertTrue(report["ready"])
        self.assertEqual(report["invariant_check_count"], 90)
        self.assertEqual(report["invariant_violation_count"], 0)
        self.assertTrue(all(count > 0 for count in report["cancellation_stage_counts"].values()))

class LiveHarnessContractTests(unittest.IsolatedAsyncioTestCase):
    async def test_live_harness_rejects_unbounded_explicit_request_text(self) -> None:
        with self.assertRaisesRegex(ValueError, "request text"):
            await run_live_sequential_gate(
                host="127.0.0.1",
                port=8765,
                auth_key=VoiceAuthKey(1, b"k" * 32),
                turns=1,
                request_text="x" * 257,
            )

    async def test_live_harness_observes_exact_sequential_protocol(self) -> None:
        from websockets.asyncio.server import serve
        from jarvis_voice.fakes import FakeOrchestrator
        from jarvis_voice.protocol import Envelope, new_uuid4
        from jarvis_voice.voice_auth_transport import authenticate_async_server

        auth_key = VoiceAuthKey(1, b"k" * 32)

        async def handle(websocket: object) -> None:
            fake: FakeOrchestrator | None = None
            hello_raw = await authenticate_async_server(
                websocket,
                auth_key,
                connection_id=new_uuid4(),
            )
            pending = [hello_raw]
            while True:
                if pending:
                    raw = pending.pop()
                else:
                    try:
                        raw = await websocket.recv()  # type: ignore[attr-defined]
                    except Exception:
                        return
                event = Envelope.from_json(raw)
                if fake is None:
                    fake = FakeOrchestrator(session_id=event.session_id)
                if event.type == "client.hello":
                    for reply in fake.accept(event):
                        await websocket.send(reply.to_json())  # type: ignore[attr-defined]
                elif event.type == "input.transcript.final":
                    response_id = new_uuid4()
                    ack = fake.factory.create(
                        "server.ack",
                        {"acked_event_id": event.event_id, "status": "accepted"},
                        turn_id=event.turn_id,
                        response_id=response_id,
                    )
                    await websocket.send(ack.to_json())  # type: ignore[attr-defined]
                    scripted = fake.script_response(
                        ("Ready.",),
                        locale="en-US",
                        turn_id=event.turn_id,
                        response_id=response_id,
                        interleave_speech=True,
                    )
                    for reply in scripted.events:
                        await websocket.send(reply.to_json())  # type: ignore[attr-defined]

        try:
            server = await serve(handle, "127.0.0.1", 0)
        except PermissionError:
            self.skipTest("sandbox forbids loopback socket bind")
        try:
            port = server.sockets[0].getsockname()[1]
            report = await run_live_sequential_gate(
                host="127.0.0.1",
                port=port,
                auth_key=auth_key,
                turns=3,
                timeout_seconds=2,
                include_latency_profile=True,
            )
        finally:
            server.close()
            await server.wait_closed()
        self.assertTrue(report["ready"])
        self.assertEqual(report["response_count"], 3)
        self.assertEqual(report["duplicate_publication_count"], 0)
        self.assertFalse(report["physical_speaker_playback_observed"])
        self.assertEqual(report["playback_completion_mode"], "protocol_probe")
        self.assertEqual(
            report["latency_profile_ms"]["transcript_to_first_speech_commit"]["count"],
            3,
        )
        self.assertEqual(
            set(report["latency_profile_ms"]["ack_to_first_speech_commit"]),
            {"count", "p50", "p95", "maximum"},
        )


if __name__ == "__main__":
    unittest.main()
