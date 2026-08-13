from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import _bootstrap  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "scripts/run_live_memory_crud_probe.py"
SPEC = importlib.util.spec_from_file_location("live_memory_crud_probe", PROBE)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class LiveMemoryCrudProbeSafetyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(dir="/private/tmp")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.database = self.root / "orchestrator.sqlite3"
        self.database.touch()
        self.config = self.root / "orchestrator.test.json"

    def _args(
        self,
        *,
        database: Path | None = None,
        database_path: Path | None = None,
        url: str = "ws://127.0.0.1:8865",
        control_url: str = "http://127.0.0.1:8867",
        voice_port: int = 8865,
        control_port: int = 8867,
        **overrides: object,
    ) -> argparse.Namespace:
        selected_database = database or self.database
        document: dict[str, object] = {
            "database_path": str(database_path or selected_database),
            "control_bearer_token": "synthetic-control-token",
            "voice_port": voice_port,
            "control_port": control_port,
            "memory_enabled": True,
            "memory_shadow_enabled": False,
            "research_enabled": False,
            "cloud_enabled": False,
            "tools_enabled": False,
        }
        document.update(overrides)
        self.config.write_text(json.dumps(document), encoding="utf-8")
        return argparse.Namespace(
            url=url,
            control_url=control_url,
            config=self.config,
            database=selected_database,
            timeout=0.01,
        )

    def test_exact_disposable_target_is_accepted(self) -> None:
        MODULE._validate_disposable_target(self._args())

    def test_main_like_endpoints_never_reach_cleanup_mutations(self) -> None:
        args = self._args(
            url="ws://127.0.0.1:8767",
            control_url="http://127.0.0.1:8765",
            voice_port=8767,
            control_port=8765,
        )
        with (
            patch.object(MODULE, "_read_only") as read_only,
            patch.object(MODULE, "_cancel_task") as cancel,
            patch.object(MODULE, "_delete_conversation") as delete,
        ):
            self.assertFalse(MODULE._best_effort_cleanup(args))
        read_only.assert_not_called()
        cancel.assert_not_called()
        delete.assert_not_called()

    def test_config_database_mismatch_never_reaches_cleanup_mutations(self) -> None:
        other_database = self.root / "other.sqlite3"
        other_database.touch()
        args = self._args(database_path=other_database)
        with (
            patch.object(MODULE, "_read_only") as read_only,
            patch.object(MODULE, "_cancel_task") as cancel,
            patch.object(MODULE, "_delete_conversation") as delete,
        ):
            self.assertFalse(MODULE._best_effort_cleanup(args))
        read_only.assert_not_called()
        cancel.assert_not_called()
        delete.assert_not_called()

    def test_unsafe_capability_config_never_reaches_cleanup_mutations(self) -> None:
        args = self._args(tools_enabled=True)
        with (
            patch.object(MODULE, "_read_only") as read_only,
            patch.object(MODULE, "_cancel_task") as cancel,
            patch.object(MODULE, "_delete_conversation") as delete,
        ):
            self.assertFalse(MODULE._best_effort_cleanup(args))
        read_only.assert_not_called()
        cancel.assert_not_called()
        delete.assert_not_called()

    def test_validated_cleanup_deletes_only_the_disposable_conversation(self) -> None:
        args = self._args()
        read_results = [
            [{"conversation_id": "synthetic-conversation", "current_epoch": 7}],
            [],
            [],
        ]
        with (
            patch.object(MODULE, "_read_only", side_effect=read_results),
            patch.object(MODULE, "_cancel_task") as cancel,
            patch.object(MODULE, "_delete_conversation", return_value=204) as delete,
        ):
            self.assertTrue(MODULE._best_effort_cleanup(args))
        cancel.assert_not_called()
        delete.assert_called_once_with(
            args.control_url,
            args.config,
            "synthetic-conversation",
            7,
        )

    def test_closed_failure_codes_never_echo_unknown_exception_text(self) -> None:
        for code in MODULE._CLOSED_FAILURE_CODES:
            with self.subTest(code=code):
                self.assertEqual(MODULE._closed_failure_code(RuntimeError(code)), code)
        self.assertEqual(
            MODULE._closed_failure_code(
                RuntimeError("private transcript and bearer token must not escape")
            ),
            "probe_failed",
        )

    def test_main_emits_content_free_closed_failure_report(self) -> None:
        args = self._args()
        parser = unittest.mock.Mock()
        parser.parse_args.return_value = args
        output = io.StringIO()
        sensitive = "private transcript and bearer token must not escape"
        with (
            patch.object(MODULE, "parser", return_value=parser),
            patch.object(
                MODULE,
                "_validate_disposable_inputs",
                side_effect=RuntimeError(sensitive),
            ),
            patch.object(
                MODULE, "run", new=AsyncMock(side_effect=AssertionError("must not run"))
            ) as run,
            patch.object(MODULE, "_best_effort_cleanup") as cleanup,
            contextlib.redirect_stdout(output),
        ):
            self.assertEqual(MODULE.main(), 2)
        run.assert_not_called()
        cleanup.assert_not_called()
        rendered = output.getvalue()
        self.assertNotIn(sensitive, rendered)
        self.assertNotIn("synthetic-control-token", rendered)
        self.assertNotIn(str(self.database), rendered)
        self.assertEqual(
            json.loads(rendered),
            {
                "cleanup_attempted": False,
                "cleanup_succeeded": False,
                "contains_user_content": False,
                "error_code": "probe_failed",
                "probe_version": "jarvis.live-memory-crud.v1",
                "ready": False,
            },
        )

    def test_preexisting_conversation_never_arms_cleanup(self) -> None:
        args = self._args()
        authority_tables = (
            "approvals",
            "capability_reservations",
            "cloud_dispatch_permits",
            "cloud_provider_receipts",
            "research_runs",
            "scheduler_jobs",
            "scheduler_occurrences",
            "scheduler_proposals",
            "tool_executions",
            "memory_records",
        )
        connection = sqlite3.connect(self.database)
        try:
            for table in authority_tables:
                connection.execute(f"CREATE TABLE {table} (synthetic_id INTEGER)")
            connection.execute("CREATE TABLE canonical_turns (effects_json TEXT)")
            connection.execute("CREATE TABLE conversations (synthetic_id INTEGER)")
            connection.execute("CREATE TABLE tasks (synthetic_id INTEGER)")
            connection.execute("INSERT INTO conversations VALUES (1)")
            connection.commit()
        finally:
            connection.close()

        parser = unittest.mock.Mock()
        parser.parse_args.return_value = args
        output = io.StringIO()
        with (
            patch.object(MODULE, "parser", return_value=parser),
            patch.object(
                MODULE, "run", new=AsyncMock(side_effect=AssertionError("must not run"))
            ) as run,
            patch.object(MODULE, "_best_effort_cleanup") as cleanup,
            patch.object(MODULE, "_cancel_task") as cancel,
            patch.object(MODULE, "_delete_conversation") as delete,
            contextlib.redirect_stdout(output),
        ):
            self.assertEqual(MODULE.main(), 2)
        run.assert_not_called()
        cleanup.assert_not_called()
        cancel.assert_not_called()
        delete.assert_not_called()
        self.assertEqual(
            json.loads(output.getvalue()),
            {
                "cleanup_attempted": False,
                "cleanup_succeeded": False,
                "contains_user_content": False,
                "error_code": "probe_failed",
                "probe_version": "jarvis.live-memory-crud.v1",
                "ready": False,
            },
        )

    def test_main_does_not_attempt_cleanup_for_main_like_target(self) -> None:
        args = self._args(
            url="ws://127.0.0.1:8767",
            control_url="http://127.0.0.1:8765",
            voice_port=8767,
            control_port=8765,
        )
        parser = unittest.mock.Mock()
        parser.parse_args.return_value = args
        output = io.StringIO()
        with (
            patch.object(MODULE, "parser", return_value=parser),
            patch.object(
                MODULE,
                "run",
                new=AsyncMock(side_effect=RuntimeError("probe_target_unsafe")),
            ),
            patch.object(MODULE, "_best_effort_cleanup") as cleanup,
            patch.object(MODULE, "_cancel_task") as cancel,
            patch.object(MODULE, "_delete_conversation") as delete,
            contextlib.redirect_stdout(output),
        ):
            self.assertEqual(MODULE.main(), 2)
        cleanup.assert_not_called()
        cancel.assert_not_called()
        delete.assert_not_called()
        self.assertEqual(json.loads(output.getvalue())["cleanup_attempted"], False)


if __name__ == "__main__":
    unittest.main()
