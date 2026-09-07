from __future__ import annotations

from dataclasses import replace
import http.client
from http.server import ThreadingHTTPServer
import json
import socket
import threading
import unittest
import uuid

import _bootstrap  # noqa: F401
from helpers import StoreCase
from jarvis_orchestrator.config import OrchestratorConfig
from jarvis_orchestrator.control import ControlApplication, handler_factory
from jarvis_orchestrator.scheduler_contract import PackagedTimezoneProvider
from jarvis_orchestrator.scheduler_crypto import InMemorySchedulerKeyProvider, SchedulerContentCipher
from jarvis_orchestrator.scheduler_store import SchedulerStore
from jarvis_orchestrator.service import OrchestratorService
from test_service_control_continuation import FakeModel


class ControlRequestBoundaryTests(StoreCase, unittest.TestCase):
    def setUp(self):
        self.store = self.make_store()
        config = OrchestratorConfig.from_test_mapping({
            "database_path": str(self.store.path), "control_bearer_token": "synthetic-control",
            "inference_bearer_token": "synthetic-inference",
        })
        service = OrchestratorService(config, self.store, FakeModel())
        service.scheduler = SchedulerStore(self.store,
            cipher=SchedulerContentCipher(InMemorySchedulerKeyProvider()), zones=PackagedTimezoneProvider())
        application = ControlApplication(config, service, business_admission=lambda: True)
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), handler_factory(application))
        application.config = replace(config, control_port=self.server.server_port)
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
        self.thread.start()
        self.addCleanup(self.close_server)

    def close_server(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=1)
        self.assertFalse(self.thread.is_alive())

    def request(self, method, path, headers=(), body=b""):
        lines = [f"{method} {path} HTTP/1.0", f"Host: 127.0.0.1:{self.server.server_port}",
                 "Authorization: Bearer synthetic-control", "Connection: close", *headers, "", ""]
        with socket.create_connection(self.server.server_address, timeout=1) as connection:
            connection.sendall("\r\n".join(lines).encode("ascii") + body)
            connection.shutdown(socket.SHUT_WR)
            with http.client.HTTPResponse(connection) as response:
                response.begin()
                payload = response.read()
                return response.status, json.loads(payload) if payload else None

    def mutation_headers(self, length):
        return ["Content-Type: application/json", f"Content-Length: {length}",
                f"Idempotency-Key: {uuid.uuid4()}", "If-Match-Epoch: 0"]

    def test_truncated_or_transfer_encoded_post_never_creates_conversation(self):
        for extra, length in (([], 10), (["Transfer-Encoding: chunked"], 2),
                              (["Transfer-Encoding: identity"], 2)):
            with self.subTest(extra=extra, length=length):
                status, body = self.request("POST", "/v1/conversations",
                                            self.mutation_headers(length) + extra, b"{}")
                self.assertEqual((status, body["error"]["code"] if "error" in body else None),
                                 (400, "invalid_request"))
                self.assertIsNone(self.store.active_conversation())

    def test_complete_post_still_creates_conversation(self):
        status, _body = self.request("POST", "/v1/conversations", self.mutation_headers(2), b"{}")
        self.assertEqual(status, 201)
        self.assertIsNotNone(self.store.active_conversation())

    def test_transfer_encoding_is_rejected_for_reads_and_deletes(self):
        self.request("POST", "/v1/conversations", self.mutation_headers(2), b"{}")
        conversation = self.store.active_conversation()
        for method, path in (("GET", "/v1/conversations/active"),
                             ("DELETE", f"/v1/conversations/{conversation['conversation_id']}")):
            with self.subTest(method=method):
                headers = [f"Idempotency-Key: {uuid.uuid4()}", f"If-Match-Epoch: {conversation['epoch']}",
                           "Content-Length: 0", "Transfer-Encoding: chunked"]
                status, body = self.request(method, path, headers)
                self.assertEqual(status, 400)
                self.assertEqual(body["error"]["code"], "invalid_request")
                self.assertEqual(self.store.active_conversation()["conversation_id"], conversation["conversation_id"])

    def test_invalid_scheduler_canonical_values_are_client_errors(self):
        for raw in (b"NaN", b"1.5", b'{"input":{},"output":{"x":0.5}}', b'{"\\u00e9":1}'):
            with self.subTest(raw=raw):
                status, body = self.request("POST", "/v1/scheduler/proposals",
                                            self.mutation_headers(len(raw)), raw)
                self.assertEqual((status, body["error"]["code"]), (400, "invalid_request"))
                self.assertEqual(self.store._connection.execute("SELECT COUNT(*) FROM scheduler_inbox").fetchone()[0], 0)

    def test_malformed_request_target_is_a_client_error_for_every_handler(self):
        for method in ("GET", "POST", "PUT", "DELETE"):
            with self.subTest(method=method):
                status, body = self.request(method, "http://[", self.mutation_headers(2), b"{}")
                self.assertEqual((status, body["error"]["code"]), (400, "invalid_request"))
                self.assertIsNone(self.store.active_conversation())

    def test_scheduler_ledger_rejects_out_of_range_cursor_as_client_error(self):
        for sequence in ((1 << 63) - 1, 1 << 63, 10 ** 100):
            with self.subTest(sequence=sequence):
                status, body = self.request("GET", f"/v1/scheduler/ledger?after_seq={sequence}")
                if sequence == (1 << 63) - 1:
                    self.assertEqual((status, body), (200, {"ledger": []}))
                else:
                    self.assertEqual((status, body["error"]["code"]), (400, "invalid_request"))


if __name__ == "__main__":
    unittest.main()
