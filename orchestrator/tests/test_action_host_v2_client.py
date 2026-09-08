from __future__ import annotations

import asyncio
import hashlib
import time
from unittest.mock import patch
import os
import socket
import struct
import tempfile
import threading
import unittest
import uuid
from pathlib import Path

import _bootstrap  # noqa: F401

from jarvis_orchestrator.action_host_protocol import ActionHostProtocolError, encode_frame, hkdf_extract
from jarvis_orchestrator.action_host_v2_client import ActionHostV2Client
from jarvis_orchestrator.action_host_v2_protocol import (
    MAX_ENCRYPTED_FRAME_BYTES,
    MAX_HANDSHAKE_FRAME_BYTES,
    ClientResponseV2,
    DirectionV2,
    HostChallengeV2,
    HostReadyV2,
    SessionKeysV2,
    decode_handshake_frame_v2,
)
from jarvis_orchestrator.jcs import jcs_bytes


class Keys:
    def __init__(self, key: bytes) -> None:
        self.key = key

    def load(self, version: int) -> bytes:
        if version != 1:
            raise KeyError(version)
        return self.key


def read_exact(sock: socket.socket, length: int) -> bytes:
    value = b""
    while len(value) < length:
        chunk = sock.recv(length - len(value))
        if not chunk:
            raise EOFError
        value += chunk
    return value


def read_frame(sock: socket.socket, maximum: int) -> bytes:
    prefix = read_exact(sock, 4)
    length = struct.unpack(">I", prefix)[0]
    if not 0 < length <= maximum:
        raise ValueError("frame")
    return prefix + read_exact(sock, length)


class FakeHost:
    def __init__(self, root: Path, key: bytes, token: bytes) -> None:
        self.socket_path = root / "host.sock"
        self.key = key
        self.token = token
        self.host_id = str(uuid.uuid4())
        self.result = None
        self.receipts = []
        self.acked = False
        self.ready = threading.Event()
        self.stop = threading.Event()
        self.failure: Exception | None = None
        self.thread = threading.Thread(target=self.run, daemon=True)

    def start(self) -> None:
        self.thread.start()
        if not self.ready.wait(2):
            raise RuntimeError("host did not start")

    def close(self) -> None:
        self.stop.set()
        try:
            wake = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            wake.connect(os.fspath(self.socket_path))
            wake.close()
        except OSError:
            pass
        self.thread.join(2)
        if self.failure is not None:
            raise self.failure

    def run(self) -> None:
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            listener.bind(os.fspath(self.socket_path))
            os.chmod(self.socket_path, 0o600)
            listener.listen(4)
            listener.settimeout(0.1)
            self.ready.set()
            while not self.stop.is_set():
                try:
                    connection, _ = listener.accept()
                except TimeoutError:
                    continue
                try:
                    self.handle(connection)
                except (EOFError, BrokenPipeError, ConnectionResetError):
                    pass
                except OSError:
                    if not self.stop.is_set():
                        raise
                finally:
                    connection.close()
        except Exception as exc:
            self.failure = exc
            self.ready.set()
        finally:
            listener.close()

    def handle(self, connection: socket.socket) -> None:
        challenge = HostChallengeV2.create(
            auth_key=self.key,
            instance_token=self.token,
            host_instance_id=self.host_id,
            key_version=1,
        )
        connection.sendall(
            encode_frame(challenge.to_dict(), maximum=MAX_HANDSHAKE_FRAME_BYTES)
        )
        response = ClientResponseV2.from_dict(
            decode_handshake_frame_v2(
                read_frame(connection, MAX_HANDSHAKE_FRAME_BYTES)
            )
        )
        session_id = str(uuid.uuid4())
        keys = SessionKeysV2.derive(
            auth_key=self.key,
            instance_token=self.token,
            challenge=challenge,
            response=response,
            session_id=session_id,
        )
        ready = HostReadyV2.create(
            k_boot=hkdf_extract(self.token, self.key),
            session_id=session_id,
            host_instance_id=self.host_id,
        )
        connection.sendall(
            encode_frame(ready.to_dict(), maximum=MAX_HANDSHAKE_FRAME_BYTES)
        )
        receiver = keys.codec(DirectionV2.C2H)
        sender = keys.codec(DirectionV2.H2C)
        while True:
            message = receiver.decrypt(read_frame(connection, MAX_ENCRYPTED_FRAME_BYTES))
            if message["type"] == "action.execute":
                self._execute(connection, sender, message["payload"])
                return
            if message["type"] == "action.result.query":
                connection.sendall(
                    sender.encrypt(
                        {"type": "action.result.snapshot", "payload": {"result": self.result}}
                    )
                )
            elif message["type"] == "action.receipt.query":
                connection.sendall(
                    sender.encrypt(
                        {
                            "type": "action.receipt.snapshot",
                            "payload": {
                                "execution_id": self.result["execution_id"],
                                "idempotency_key": self.result["idempotency_key"],
                                "receipts": self.receipts,
                            },
                        }
                    )
                )
                return
            elif message["type"] == "action.result.ack":
                if (
                    message["payload"]["result_event_id"] != self.result["result_event_id"]
                    or message["payload"]["result_sha256"] != self.result["result_sha256"]
                ):
                    raise AssertionError("conflicting ack")
                self.acked = True
            elif message["type"] == "action.ping":
                connection.sendall(
                    sender.encrypt({"type": "action.pong", "payload": message["payload"]})
                )
                return

    def _receipt(self, execute, stage, result_sha=""):
        return {
            "request_id": execute["request_id"],
            "execution_id": execute["execution_id"],
            "idempotency_key": execute["idempotency_key"],
            "stage": stage,
            "observed_at_utc": "2026-08-11T18:00:00.000Z",
            "adapter_status": "RETURNED" if stage == "VERIFICATION_COMPLETE" else "PENDING",
            "os_receipt_sha256": "c" * 64 if stage != "HOST_RECEIVED" else "",
            "result_sha256": result_sha,
            "verification": "MATCHED" if stage == "VERIFICATION_COMPLETE" else "NOT_APPLICABLE",
            "reason_code": "ok",
        }

    def _execute(self, connection, sender, execute):
        payload = {
            "app_id": "finder",
            "bundle_id": "com.apple.finder",
            "request_observed_at_utc": "2026-08-11T18:00:00.000Z",
        }
        digest = hashlib.sha256(jcs_bytes(payload)).hexdigest()
        self.result = {
            "result_event_id": str(uuid.uuid4()),
            "request_id": execute["request_id"],
            "execution_id": execute["execution_id"],
            "tool_call_id": execute["tool_call_id"],
            "permit_id": execute["permit_id"],
            "idempotency_key": execute["idempotency_key"],
            "tool_id": execute["tool_id"],
            "tool_version": execute["tool_version"],
            "result_schema_sha256": "d" * 64,
            "registry_sha256": execute["registry_sha256"],
            "status": "VERIFIED",
            "verification": "EXACT_READBACK",
            "reason_code": "ok",
            "observed_at_utc": "2026-08-11T18:00:00.000Z",
            "result_payload": payload,
            "result_sha256": digest,
        }
        self.receipts = [
            self._receipt(execute, "HOST_RECEIVED"),
            self._receipt(execute, "OS_CALL_ENTERING"),
            self._receipt(execute, "OS_CALL_RETURNED"),
            self._receipt(execute, "VERIFICATION_COMPLETE", digest),
        ]
        for receipt in self.receipts:
            connection.sendall(sender.encrypt({"type": "action.receipt", "payload": receipt}))
        connection.sendall(sender.encrypt({"type": "action.result", "payload": self.result}))


class ActionHostV2ClientTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        os.chmod(self.root, 0o700)
        self.token = bytes(reversed(range(32)))
        self.token_path = self.root / "token"
        self.token_path.write_bytes(self.token)
        os.chmod(self.token_path, 0o600)
        self.key = bytes(range(32))
        self.host = FakeHost(self.root, self.key, self.token)
        self.host.start()
        self.client = ActionHostV2Client(
            socket_path=self.host.socket_path,
            token_path=self.token_path,
            auth_keys=Keys(self.key),
            timeout_seconds=2,
        )

    async def asyncTearDown(self) -> None:
        self.host.close()
        self.temp.cleanup()

    def execute_payload(self):
        arguments = {"app_id": "finder"}
        return {
            "request_id": str(uuid.uuid4()),
            "execution_id": str(uuid.uuid4()),
            "tool_call_id": str(uuid.uuid4()),
            "permit_id": str(uuid.uuid4()),
            "idempotency_key": str(uuid.uuid4()),
            "tool_id": "app.open@1",
            "tool_version": 1,
            "tool_schema_sha256": "a" * 64,
            "registry_sha256": "b" * 64,
            "policy_sha256": "c" * 64,
            "scheduler_contract_sha256": "d" * 64,
            "user_intent_binding_sha256": "e" * 64,
            "arguments_sha256": hashlib.sha256(jcs_bytes(arguments)).hexdigest(),
            "payload": arguments,
            "deadline_utc": "2026-08-11T18:01:00.000Z",
        }

    async def test_execute_query_replay_and_durable_ack(self) -> None:
        executed = await self.client.execute(self.execute_payload())
        self.assertEqual(executed.outcome.status.value, "VERIFIED")
        self.assertFalse(self.host.acked)
        replay = await self.client.query_result(
            executed.execution_id, executed.idempotency_key
        )
        self.assertEqual(replay.result_event_id, executed.result_event_id)
        self.assertEqual(replay.result_sha256, executed.result_sha256)
        await self.client.acknowledge(executed)
        self.assertTrue(self.host.acked)



    async def test_cancellation_closes_socket_before_returning(self):
        entered = threading.Event()
        closed = threading.Event()
        def stalled_host(connection):
            entered.set()
            if connection.recv(1) == b"":
                closed.set()
        with patch.object(self.host, "handle", side_effect=stalled_host):
            task = asyncio.create_task(self.client.execute(self.execute_payload()))
            try:
                self.assertTrue(await asyncio.to_thread(entered.wait, 1))
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
                self.assertTrue(await asyncio.to_thread(closed.wait, 0.2))
            finally:
                task.cancel()

    async def test_drip_fed_handshake_obeys_total_timeout(self):
        self.client.timeout_seconds = 0.08
        def drip_host(connection):
            try:
                for byte in struct.pack(">I", 100) + b"x" * 20:
                    connection.sendall(bytes([byte]))
                    time.sleep(0.03)
            except OSError:
                pass
        with patch.object(self.host, "handle", side_effect=drip_host):
            start = time.monotonic()
            with self.assertRaises(TimeoutError):
                await self.client.execute(self.execute_payload())
            self.assertLess(time.monotonic() - start, 0.4)


    async def test_query_rejects_result_from_a_different_execution(self):
        result = await self.client.execute(self.execute_payload())
        with self.assertRaisesRegex(ActionHostProtocolError, "host.result_conflict"):
            await self.client.query_result(str(uuid.uuid4()), result.idempotency_key)

if __name__ == "__main__":
    unittest.main()
