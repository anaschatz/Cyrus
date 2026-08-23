from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import sqlite3
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import _bootstrap  # noqa: F401

from jarvis_orchestrator.gateway import MAX_PENDING_AUTHENTICATIONS, VoiceGateway
from jarvis_orchestrator.domain import OrchestratorError
from jarvis_orchestrator.service import OrchestratorService
from jarvis_voice.protocol import EventFactory
from jarvis_voice.voice_auth import VoiceAuthKey, create_client_response


AUTH_KEY = VoiceAuthKey(1, bytes(range(32)))


class GuardedStore:
    def __init__(self) -> None:
        self.business_accessed = False

    def active_conversation(self):
        self.business_accessed = True
        return None

    def claim_cloud_presentation_owner(self, _session_id: str):
        return None


class GuardedService:
    def __init__(
        self,
        *,
        cloud_enabled: bool = False,
        cloud_model: object | None = None,
        cloud_endpoints: tuple[object, ...] = (),
    ) -> None:
        self.config = SimpleNamespace(
            cloud_enabled=cloud_enabled,
            voice_scheduler_compatibility_sha256="",
            voice_scheduler_callback_qualification_sha256="",
        )
        self.store = GuardedStore()
        self.scheduler = None
        self.tool_store = None
        self.task_profiler = object() if cloud_enabled else None
        self.cloud_model = cloud_model
        self.cloud_endpoints = cloud_endpoints
        self.attached: list[str] = []
        self.attach_options: list[dict[str, object]] = []
        self.detached: list[str] = []

    def cloud_approval_available(self) -> bool:
        return OrchestratorService.cloud_approval_available(self)  # type: ignore[arg-type]

    def attach_client(self, connection_id: str, **kwargs: object) -> None:
        self.attached.append(connection_id)
        self.attach_options.append(kwargs)

    def detach_client(self, connection_id: str) -> None:
        self.detached.append(connection_id)


class AuthWebSocket:
    remote_address = ("127.0.0.1", 50000)

    def __init__(
        self,
        hello: str,
        *,
        mode: str = "valid",
        signed_hello: str | None = None,
        request_path: str = "/",
    ) -> None:
        self.hello = hello
        self.signed_hello = signed_hello or hello
        self.mode = mode
        self.sent: list[str] = []
        self.recv_count = 0
        self.closed: list[tuple[int, str]] = []
        self.request = SimpleNamespace(path=request_path)

    async def send(self, raw: str) -> None:
        self.sent.append(raw)

    async def recv(self) -> str:
        if self.mode == "legacy":
            self.recv_count += 1
            return self.hello
        if self.recv_count == 0:
            self.recv_count += 1
            challenge_raw = self.sent[0]
            if self.mode == "wrong_key":
                challenge = json.loads(challenge_raw)
                random_32 = base64.urlsafe_b64encode(bytes(32)).rstrip(b"=").decode()
                return json.dumps(
                    {
                        "client_mac": random_32,
                        "client_nonce": random_32,
                        "connection_id": challenge["connection_id"],
                        "hello_sha256": hashlib.sha256(
                            self.signed_hello.encode()
                        ).hexdigest(),
                        "key_version": challenge["key_version"],
                        "protocol": "jarvis.voice-auth/1",
                        "server_nonce": challenge["server_nonce"],
                        "type": "client.response",
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
            return create_client_response(
                AUTH_KEY,
                challenge_raw,
                self.signed_hello,
            ).to_json()
        if self.recv_count == 1:
            self.recv_count += 1
            return self.hello
        raise StopAsyncIteration

    def __aiter__(self):
        return self

    async def __anext__(self) -> str:
        raise StopAsyncIteration

    async def close(self, *, code: int, reason: str) -> None:
        self.closed.append((code, reason))


class VoiceGatewayAuthenticationTests(unittest.TestCase):
    def test_service_error_reporter_emits_only_bounded_diagnostics(self) -> None:
        reports: list[tuple[str, int | None]] = []
        gateway = VoiceGateway(  # type: ignore[arg-type]
            GuardedService(),
            auth_key=AUTH_KEY,
            service_error_reporter=lambda code, detail: reports.append((code, detail)),
        )

        gateway._report_service_error(
            "transcript_admission",
            sqlite3.IntegrityError("private transcript must not escape"),
        )
        gateway._report_service_error(
            "voice_event",
            OrchestratorError("private error detail", status=409),
        )

        self.assertEqual(
            reports,
            [
                ("gateway_service_error_transcript_admission_sqlite_integrity", None),
                ("gateway_service_error_voice_event_orchestrator", 409),
            ],
        )
        self.assertNotIn("private", repr(reports))

    def hello(self, *, extra_capability: str | None = None) -> str:
        capabilities = ["speech_commits", "speech_done", "output_epochs"]
        if extra_capability is not None:
            capabilities.append(extra_capability)
        return EventFactory().create(
            "client.hello",
            {"capabilities": capabilities, "protocol": "jarvis.voice/1"},
        ).to_json()

    def run_gateway(
        self,
        websocket: AuthWebSocket,
        *,
        service: GuardedService | None = None,
        admitted: bool = True,
    ) -> GuardedService:
        service = service or GuardedService()
        gateway = VoiceGateway(  # type: ignore[arg-type]
            service,
            auth_key=AUTH_KEY,
            business_admission=lambda: admitted,
        )
        asyncio.run(gateway.handle(websocket))
        return service

    def test_authenticated_voice_business_is_rejected_before_runtime_barrier(
        self,
    ) -> None:
        websocket = AuthWebSocket(self.hello())
        service = self.run_gateway(websocket, admitted=False)
        documents = [json.loads(raw) for raw in websocket.sent]

        self.assertEqual(
            [item["type"] for item in documents],
            ["server.challenge", "server.ready"],
        )
        self.assertEqual(websocket.closed, [(1013, "runtime_not_ready")])
        self.assertEqual(service.attached, [])
        self.assertFalse(service.store.business_accessed)

    def test_legacy_unauthenticated_hello_is_rejected_before_business_state(self) -> None:
        websocket = AuthWebSocket(self.hello(), mode="legacy")
        service = self.run_gateway(websocket)

        self.assertEqual(websocket.closed, [(1008, "authentication_failed")])
        self.assertEqual(service.attached, [])
        self.assertFalse(service.store.business_accessed)

    def test_wrong_client_mac_is_rejected_before_business_state(self) -> None:
        websocket = AuthWebSocket(self.hello(), mode="wrong_key")
        service = self.run_gateway(websocket)

        self.assertEqual(websocket.closed, [(1008, "authentication_failed")])
        self.assertEqual(service.attached, [])
        self.assertFalse(service.store.business_accessed)

    def test_authenticated_response_cannot_be_rebound_to_changed_hello(self) -> None:
        signed = self.hello()
        changed = self.hello(extra_capability="jarvis.cloud-approval/1")
        websocket = AuthWebSocket(changed, signed_hello=signed)
        service = self.run_gateway(websocket)

        self.assertEqual(websocket.closed, [(1008, "authentication_failed")])
        self.assertEqual(service.attached, [])
        self.assertFalse(service.store.business_accessed)

    def test_valid_mutual_authentication_precedes_voice_ready_and_attach(self) -> None:
        websocket = AuthWebSocket(self.hello())
        service = self.run_gateway(websocket)
        documents = [json.loads(raw) for raw in websocket.sent]

        self.assertEqual(documents[0]["type"], "server.challenge")
        self.assertEqual(documents[1]["protocol"], "jarvis.voice-auth/1")
        self.assertEqual(documents[1]["type"], "server.ready")
        self.assertEqual(documents[2]["type"], "server.ready")
        self.assertNotIn("protocol", documents[2])
        self.assertEqual(len(service.attached), 1)
        self.assertEqual(service.detached, service.attached)
        self.assertTrue(service.store.business_accessed)
        self.assertEqual(websocket.closed, [])

    def test_client_cloud_support_does_not_enable_disabled_server_feature(self) -> None:
        websocket = AuthWebSocket(self.hello(extra_capability="jarvis.cloud-approval/1"))
        service = self.run_gateway(
            websocket,
            service=GuardedService(
                cloud_enabled=False,
                cloud_model=object(),
                cloud_endpoints=(object(),),
            ),
        )
        documents = [json.loads(raw) for raw in websocket.sent]

        self.assertEqual([item["type"] for item in documents], [
            "server.challenge",
            "server.ready",
            "server.ready",
        ])
        self.assertIsNone(service.attach_options[0]["cloud_session_id"])

    def test_cloud_extension_requires_effective_adapter_and_endpoints(self) -> None:
        for service in (
            GuardedService(cloud_enabled=True, cloud_model=None, cloud_endpoints=(object(),)),
            GuardedService(cloud_enabled=True, cloud_model=object(), cloud_endpoints=()),
        ):
            with self.subTest(
                adapter=service.cloud_model is not None,
                endpoints=bool(service.cloud_endpoints),
            ):
                websocket = AuthWebSocket(
                    self.hello(extra_capability="jarvis.cloud-approval/1")
                )
                self.run_gateway(websocket, service=service)
                documents = [json.loads(raw) for raw in websocket.sent]

                self.assertNotIn("cloud.approval.ready", [item["type"] for item in documents])
                self.assertIsNone(service.attach_options[0]["cloud_session_id"])

    def test_effective_cloud_feature_negotiates_existing_extension(self) -> None:
        websocket = AuthWebSocket(self.hello(extra_capability="jarvis.cloud-approval/1"))
        service = self.run_gateway(
            websocket,
            service=GuardedService(
                cloud_enabled=True,
                cloud_model=object(),
                cloud_endpoints=(object(),),
            ),
        )
        documents = [json.loads(raw) for raw in websocket.sent]

        self.assertIn("cloud.approval.ready", [item["type"] for item in documents])
        self.assertEqual(
            service.attach_options[0]["cloud_session_id"],
            documents[2]["session_id"],
        )

    def test_auth_probe_never_enters_business_protocol(self) -> None:
        from jarvis_voice.voice_auth_transport import (
            AUTH_PROBE_FRAME,
            AUTH_PROBE_PATH,
            AUTH_PROBE_SUCCESS_FRAME,
        )

        websocket = AuthWebSocket(AUTH_PROBE_FRAME, request_path=AUTH_PROBE_PATH)
        service = self.run_gateway(websocket)
        documents = [json.loads(raw) for raw in websocket.sent]

        self.assertEqual(documents[0]["type"], "server.challenge")
        self.assertEqual(documents[1]["type"], "server.ready")
        self.assertEqual(websocket.sent[2], AUTH_PROBE_SUCCESS_FRAME)
        self.assertEqual(service.attached, [])
        self.assertFalse(service.store.business_accessed)

    def test_invalid_auth_probe_has_no_success_ack_or_business_access(self) -> None:
        from jarvis_voice.voice_auth_transport import AUTH_PROBE_PATH

        websocket = AuthWebSocket(
            '{"protocol":"jarvis.voice-auth-probe/1","unexpected":true}',
            request_path=AUTH_PROBE_PATH,
        )
        service = self.run_gateway(websocket)

        self.assertEqual(len(websocket.sent), 2)
        self.assertEqual(
            websocket.closed,
            [(1008, "authentication_probe_invalid")],
        )
        self.assertEqual(service.attached, [])
        self.assertFalse(service.store.business_accessed)

    def test_connection_id_failure_releases_authentication_capacity(self) -> None:
        service = GuardedService()
        gateway = VoiceGateway(service, auth_key=AUTH_KEY)  # type: ignore[arg-type]
        websocket = AuthWebSocket(self.hello())

        async def fail_after_admission() -> None:
            with patch(
                "jarvis_orchestrator.gateway.new_uuid4",
                side_effect=OSError("entropy unavailable"),
            ):
                with self.assertRaises(OSError):
                    await gateway.handle(websocket)
            for _ in range(MAX_PENDING_AUTHENTICATIONS):
                await asyncio.wait_for(gateway._auth_slots.acquire(), timeout=0.1)
            for _ in range(MAX_PENDING_AUTHENTICATIONS):
                gateway._auth_slots.release()

        asyncio.run(fail_after_admission())

    def test_unauthenticated_connection_capacity_is_bounded(self) -> None:
        service = GuardedService()
        gateway = VoiceGateway(service, auth_key=AUTH_KEY)  # type: ignore[arg-type]
        websocket = AuthWebSocket(self.hello())

        async def saturated() -> None:
            for _ in range(MAX_PENDING_AUTHENTICATIONS):
                await gateway._auth_slots.acquire()
            try:
                await gateway.handle(websocket)
            finally:
                for _ in range(MAX_PENDING_AUTHENTICATIONS):
                    gateway._auth_slots.release()

        asyncio.run(saturated())

        self.assertEqual(websocket.sent, [])
        self.assertEqual(websocket.closed, [(1013, "authentication_capacity")])
        self.assertEqual(service.attached, [])
        self.assertFalse(service.store.business_accessed)


if __name__ == "__main__":
    unittest.main()
