from __future__ import annotations

import io
import tempfile
import unittest
from email.message import Message
from pathlib import Path

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
        config = OrchestratorConfig.from_mapping({
            "database_path": str(path),
            "control_bearer_token": "control-private",
            "inference_bearer_token": "inference-private",
        })
        store = OrchestratorStore(path)
        self.addCleanup(store.close)
        service = OrchestratorService(config, store, BudgetedModelPort(store, FakeModel()))
        self.handler_type = handler_factory(ControlApplication(config, service))

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


if __name__ == "__main__":
    unittest.main()
