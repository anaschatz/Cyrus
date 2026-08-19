from __future__ import annotations

import io
import tempfile
import unittest
from email.message import Message
from pathlib import Path
from unittest.mock import patch

import _bootstrap  # noqa: F401

from jarvis_orchestrator.config import OrchestratorConfig
from jarvis_orchestrator.control import ControlApplication, handler_factory
from jarvis_orchestrator.domain import OrchestratorError
from jarvis_orchestrator.model import BudgetedModelPort
from jarvis_orchestrator.service import OrchestratorService
from jarvis_orchestrator.store import OrchestratorStore

from test_service_control_continuation import FakeModel


class ControlSecurityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        path = Path(self.temporary.name) / "state.sqlite3"
        config = OrchestratorConfig.from_test_mapping({
            "database_path": str(path),
            "control_bearer_token": "control-private",
            "inference_bearer_token": "inference-private",
        })
        store = OrchestratorStore(path)
        self.addCleanup(store.close)
        service = OrchestratorService(config, store, BudgetedModelPort(store, FakeModel()))
        self.service = service
        self.application = ControlApplication(config, service)
        self.handler_type = handler_factory(self.application)

    def handler(self, headers: list[tuple[str, str]] | None = None):
        result = self.handler_type.__new__(self.handler_type)
        result.client_address = ("127.0.0.1", 12345)
        message = Message()
        for name, value in headers or [
            ("Host", "127.0.0.1:8767"),
            ("Authorization", "Bearer control-private"),
        ]:
            message[name] = value
        result.headers = message
        return result

    def test_loopback_host_forwarded_and_bearer_checks(self) -> None:
        self.handler()._admit(authenticate=True)
        wrong_peer = self.handler()
        wrong_peer.client_address = ("::1", 12345)
        with self.assertRaisesRegex(OrchestratorError, "forbidden"):
            wrong_peer._admit(authenticate=True)
        for headers, code in (
            ([('Host', 'localhost:8767'), ('Authorization', 'Bearer control-private')], "forbidden"),
            ([('Host', '127.0.0.1:8767'), ('Authorization', 'Bearer control-private'), ('Origin', 'null')], "forbidden"),
            ([('Host', '127.0.0.1:8767'), ('Authorization', 'Bearer wrong')], "unauthenticated"),
        ):
            with self.assertRaisesRegex(OrchestratorError, code):
                self.handler(headers)._admit(authenticate=True)

    def test_mutation_preconditions_and_strict_json_body(self) -> None:
        valid = self.handler([
            ("Host", "127.0.0.1:8767"),
            ("Authorization", "Bearer control-private"),
            ("Idempotency-Key", "bfa34f61-9885-4a18-8a84-c49bb0a19e0a"),
            ("If-Match-Epoch", "7"),
            ("Content-Type", "application/json"),
            ("Content-Length", "2"),
        ])
        valid.rfile = io.BytesIO(b"{}")
        self.assertEqual(valid._mutation_headers(), ("bfa34f61-9885-4a18-8a84-c49bb0a19e0a", 7))
        self.assertEqual(valid._post_body(), {})
        missing = self.handler()
        with self.assertRaisesRegex(OrchestratorError, "precondition_required"):
            missing._mutation_headers()
        oversized = self.handler([
            ("Host", "127.0.0.1:8767"),
            ("Authorization", "Bearer control-private"),
            ("Content-Type", "application/json"),
            ("Content-Length", str(16 * 1024 + 1)),
        ])
        oversized.rfile = io.BytesIO()
        with self.assertRaisesRegex(OrchestratorError, "request_too_large"):
            oversized._post_body()

    def test_beta_session_reports_real_voice_connection_and_disabled_scope(self) -> None:
        with patch.object(
            self.service,
            "readiness_snapshot",
            wraps=self.service.readiness_snapshot,
        ) as readiness_snapshot:
            status, before = self.application.get("/v1/beta-session")
        self.assertEqual(status, 200)
        readiness_snapshot.assert_called_once_with()
        self.assertFalse(before["voice_client_connected"])
        self.assertEqual(
            before["schema_version"], "jarvis-orchestrator-beta-session/2"
        )
        self.assertEqual(
            before["effective_capabilities"],
            {
                "cloud": False,
                "memory_retrospective": False,
                "memory_shadow": False,
                "research": False,
                "tools": False,
            },
        )
        self.service.attach_client("connection-1")
        try:
            _, connected = self.application.get("/v1/beta-session")
            self.assertTrue(connected["voice_client_connected"])
        finally:
            self.service.detach_client("connection-1")

    def test_disabled_memory_is_reported_disabled_without_live_probe(self) -> None:
        with patch.object(
            self.service.store,
            "memory_crypto_readiness",
            side_effect=AssertionError("disabled memory must not probe provider"),
        ) as memory_probe:
            status, readiness = self.application.get("/v1/readiness")

        self.assertEqual(status, 200)
        self.assertFalse(readiness["memory_ready"])
        self.assertEqual(readiness["memory_state"], "DISABLED")
        memory_probe.assert_not_called()

    def test_beta_session_records_voice_topology_changes_without_ids(
        self,
    ) -> None:
        status, before = self.application.get("/v1/beta-session")
        self.assertEqual(status, 200)
        self.assertFalse(before["voice_client_connected"])
        self.assertEqual(before["voice_connection_generation"], 0)
        self.assertEqual(
            before["schema_version"],
            "jarvis-orchestrator-beta-session/2",
        )
        tracker_id = before["voice_connection_tracker_id"]
        self.assertRegex(
            tracker_id,
            r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$",
        )

        self.service.attach_client("private-connection-id")
        _, connected = self.application.get("/v1/beta-session")
        self.assertTrue(connected["voice_client_connected"])
        self.assertEqual(connected["voice_connection_generation"], 1)
        self.assertEqual(
            connected["voice_connection_tracker_id"],
            tracker_id,
        )
        self.service.attach_client("private-connection-id")
        self.service.detach_client("unknown-connection-id")
        _, unchanged = self.application.get("/v1/beta-session")
        self.assertEqual(unchanged["voice_connection_generation"], 1)

        self.service.detach_client("private-connection-id")
        _, disconnected = self.application.get("/v1/beta-session")
        self.assertFalse(disconnected["voice_client_connected"])
        self.assertEqual(disconnected["voice_connection_generation"], 2)
        self.assertNotIn("private-connection-id", str(disconnected))

        replacement = OrchestratorService(
            self.service.config,
            self.service.store,
            self.service.model,
        )
        _, replacement_tracker_id, replacement_generation = (
            replacement.voice_connection_state()
        )
        self.assertNotEqual(replacement_tracker_id, tracker_id)
        self.assertEqual(replacement_generation, 0)


if __name__ == "__main__":
    unittest.main()
