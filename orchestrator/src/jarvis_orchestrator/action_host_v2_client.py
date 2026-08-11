"""Fail-closed client for the result-bearing local Action Host v2 protocol."""

from __future__ import annotations

import asyncio
import os
import socket
import struct
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .action_host_protocol import ActionHostProtocolError, encode_frame, hkdf_extract
from .action_host_security import (
    assert_peer_uid,
    load_instance_token,
    validate_action_host_paths,
)
from .action_host_v2_protocol import (
    MAX_ENCRYPTED_FRAME_BYTES,
    MAX_HANDSHAKE_FRAME_BYTES,
    ClientResponseV2,
    DirectionV2,
    HostChallengeV2,
    HostReadyV2,
    SessionKeysV2,
    decode_handshake_frame_v2,
    result_receipt_matches,
)
from .tool_contracts import ToolOutcomeV1, ToolStatus, VerificationStatus


class ActionHostAuthKeyProvider(Protocol):
    def load(self, version: int) -> bytes: ...


@dataclass(frozen=True, slots=True)
class NativeExecutionResultV2:
    result_event_id: str
    execution_id: str
    idempotency_key: str
    result_sha256: str
    outcome: ToolOutcomeV1
    result_payload: Mapping[str, Any]
    terminal_receipt: Mapping[str, Any]


def _canonical_verification(value: str) -> VerificationStatus:
    return {
        "EXACT_READBACK": VerificationStatus.MATCHED,
        "UNAVAILABLE": VerificationStatus.UNAVAILABLE,
        "NOT_APPLICABLE": VerificationStatus.NOT_APPLICABLE,
        "MISMATCH": VerificationStatus.MISMATCH,
        "INCONCLUSIVE": VerificationStatus.INCONCLUSIVE,
    }[value]


class _ConnectionV2:
    def __init__(
        self,
        sock: socket.socket,
        *,
        c2h,
        h2c,
    ) -> None:
        self.socket = sock
        self.c2h = c2h
        self.h2c = h2c

    def close(self) -> None:
        try:
            self.socket.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self.socket.close()

    def send(self, message: Mapping[str, Any]) -> None:
        self.socket.sendall(self.c2h.encrypt(message))

    def receive(self) -> dict[str, Any]:
        return self.h2c.decrypt(_read_frame(self.socket, MAX_ENCRYPTED_FRAME_BYTES))


def _read_exact(sock: socket.socket, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            raise ActionHostProtocolError("host.receipt_unknown")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _read_frame(sock: socket.socket, maximum: int) -> bytes:
    prefix = _read_exact(sock, 4)
    length = struct.unpack(">I", prefix)[0]
    if length == 0 or length > maximum:
        raise ActionHostProtocolError("host.frame_too_large")
    return prefix + _read_exact(sock, length)


class ActionHostV2Client:
    """One action per authenticated session; reconnects are query/ACK only."""

    def __init__(
        self,
        *,
        socket_path: Path,
        token_path: Path,
        auth_keys: ActionHostAuthKeyProvider,
        timeout_seconds: float = 30.0,
        client_instance_id: str | None = None,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.socket_path = Path(socket_path)
        self.token_path = Path(token_path)
        self.auth_keys = auth_keys
        self.timeout_seconds = timeout_seconds
        self.client_instance_id = client_instance_id or str(uuid.uuid4())

    def _connect(self) -> _ConnectionV2:
        validate_action_host_paths(self.socket_path, self.token_path)
        instance_token = load_instance_token(self.token_path)
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(self.timeout_seconds)
        try:
            sock.connect(os.fspath(self.socket_path))
            assert_peer_uid(sock)
            challenge = HostChallengeV2.from_dict(
                decode_handshake_frame_v2(
                    _read_frame(sock, MAX_HANDSHAKE_FRAME_BYTES)
                )
            )
            auth_key = self.auth_keys.load(challenge.key_version)
            response = ClientResponseV2.create(
                challenge=challenge,
                auth_key=auth_key,
                instance_token=instance_token,
                client_instance_id=self.client_instance_id,
            )
            sock.sendall(
                encode_frame(response.to_dict(), maximum=MAX_HANDSHAKE_FRAME_BYTES)
            )
            ready = HostReadyV2.from_dict(
                decode_handshake_frame_v2(
                    _read_frame(sock, MAX_HANDSHAKE_FRAME_BYTES)
                )
            )
            k_boot = hkdf_extract(instance_token, auth_key)
            ready.verify(k_boot)
            if not ready.accepted or ready.host_instance_id != challenge.host_instance_id:
                raise ActionHostProtocolError(ready.reason_code)
            keys = SessionKeysV2.derive(
                auth_key=auth_key,
                instance_token=instance_token,
                challenge=challenge,
                response=response,
                session_id=ready.session_id,
            )
            return _ConnectionV2(
                sock,
                c2h=keys.codec(DirectionV2.C2H),
                h2c=keys.codec(DirectionV2.H2C),
            )
        except Exception:
            sock.close()
            raise

    @staticmethod
    def _native_result(
        result: Mapping[str, Any], receipt: Mapping[str, Any]
    ) -> NativeExecutionResultV2:
        if not result_receipt_matches(result, receipt):
            raise ActionHostProtocolError("host.result_conflict")
        try:
            outcome = ToolOutcomeV1(
                ToolStatus(result["status"]),
                _canonical_verification(result["verification"]),
                result["reason_code"],
            )
        except (KeyError, ValueError) as exc:
            raise ActionHostProtocolError("host.protocol_mismatch") from exc
        return NativeExecutionResultV2(
            result_event_id=result["result_event_id"],
            execution_id=result["execution_id"],
            idempotency_key=result["idempotency_key"],
            result_sha256=result["result_sha256"],
            outcome=outcome,
            result_payload=dict(result["result_payload"]),
            terminal_receipt=dict(receipt),
        )

    def _execute_sync(self, execute_payload: Mapping[str, Any]) -> NativeExecutionResultV2:
        connection = self._connect()
        result: Mapping[str, Any] | None = None
        terminal_receipt: Mapping[str, Any] | None = None
        try:
            connection.send({"type": "action.execute", "payload": dict(execute_payload)})
            while result is None or terminal_receipt is None:
                message = connection.receive()
                if message["type"] == "action.error":
                    raise ActionHostProtocolError(message["payload"]["reason_code"])
                if message["type"] == "action.result":
                    candidate = message["payload"]
                    if result is not None and candidate["result_sha256"] != result["result_sha256"]:
                        raise ActionHostProtocolError("host.result_conflict")
                    result = candidate
                elif message["type"] == "action.receipt":
                    candidate = message["payload"]
                    if candidate["stage"] == "VERIFICATION_COMPLETE":
                        terminal_receipt = candidate
            return self._native_result(result, terminal_receipt)
        finally:
            connection.close()

    async def execute(self, execute_payload: Mapping[str, Any]) -> NativeExecutionResultV2:
        return await asyncio.to_thread(self._execute_sync, execute_payload)

    def _query_sync(
        self, execution_id: str, idempotency_key: str
    ) -> NativeExecutionResultV2:
        connection = self._connect()
        result: Mapping[str, Any] | None = None
        terminal_receipt: Mapping[str, Any] | None = None
        query = {"execution_id": execution_id, "idempotency_key": idempotency_key}
        try:
            connection.send({"type": "action.result.query", "payload": query})
            connection.send({"type": "action.receipt.query", "payload": query})
            while result is None or terminal_receipt is None:
                message = connection.receive()
                if message["type"] == "action.error":
                    raise ActionHostProtocolError(message["payload"]["reason_code"])
                if message["type"] == "action.result.snapshot":
                    result = message["payload"]["result"]
                elif message["type"] == "action.receipt.snapshot":
                    receipts = message["payload"]["receipts"]
                    for receipt in receipts:
                        if receipt["stage"] == "VERIFICATION_COMPLETE":
                            terminal_receipt = receipt
            return self._native_result(result, terminal_receipt)
        finally:
            connection.close()

    async def query_result(
        self, execution_id: str, idempotency_key: str
    ) -> NativeExecutionResultV2:
        return await asyncio.to_thread(self._query_sync, execution_id, idempotency_key)

    def _ack_sync(self, result: NativeExecutionResultV2) -> None:
        connection = self._connect()
        ping_nonce = os.urandom(16)
        try:
            connection.send(
                {
                    "type": "action.result.ack",
                    "payload": {
                        "result_event_id": result.result_event_id,
                        "execution_id": result.execution_id,
                        "idempotency_key": result.idempotency_key,
                        "result_sha256": result.result_sha256,
                    },
                }
            )
            connection.send(
                {
                    "type": "action.ping",
                    "payload": {
                        "nonce": __import__("base64")
                        .urlsafe_b64encode(ping_nonce)
                        .rstrip(b"=")
                        .decode("ascii")
                    },
                }
            )
            message = connection.receive()
            if message != {
                "type": "action.pong",
                "payload": {
                    "nonce": __import__("base64")
                    .urlsafe_b64encode(ping_nonce)
                    .rstrip(b"=")
                    .decode("ascii")
                },
            }:
                raise ActionHostProtocolError("host.result_unknown")
        finally:
            connection.close()

    async def acknowledge(self, result: NativeExecutionResultV2) -> None:
        await asyncio.to_thread(self._ack_sync, result)

