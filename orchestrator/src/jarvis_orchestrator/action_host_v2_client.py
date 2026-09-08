"""Fail-closed client for the result-bearing local Action Host v2 protocol."""

from __future__ import annotations

import asyncio
import base64
import math
import os
import socket
import struct
import uuid
from collections.abc import Mapping
from contextlib import asynccontextmanager
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
from .tool_contracts import ToolContractError, ToolOutcomeV1, ToolStatus, VerificationStatus


class ActionHostAuthKeyProvider(Protocol):
    def load(self, version: int) -> bytes: ...


@dataclass(frozen=True, slots=True)
class NativeExecutionResultV2:
    result_event_id: str
    request_id: str
    execution_id: str
    tool_call_id: str
    permit_id: str
    idempotency_key: str
    tool_id: str
    tool_version: int
    result_schema_sha256: str
    registry_sha256: str
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

    async def send(self, message: Mapping[str, Any]) -> None:
        await asyncio.get_running_loop().sock_sendall(self.socket, self.c2h.encrypt(message))

    async def receive(self) -> dict[str, Any]:
        return self.h2c.decrypt(await _read_frame(self.socket, MAX_ENCRYPTED_FRAME_BYTES))


async def _read_exact(sock: socket.socket, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = await asyncio.get_running_loop().sock_recv(sock, remaining)
        if not chunk:
            raise ActionHostProtocolError("host.receipt_unknown")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


async def _read_frame(sock: socket.socket, maximum: int) -> bytes:
    prefix = await _read_exact(sock, 4)
    length = struct.unpack(">I", prefix)[0]
    if length == 0 or length > maximum:
        raise ActionHostProtocolError("host.frame_too_large")
    return prefix + await _read_exact(sock, length)


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
        if isinstance(timeout_seconds, bool) or not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.socket_path = Path(socket_path)
        self.token_path = Path(token_path)
        self.auth_keys = auth_keys
        self.timeout_seconds = timeout_seconds
        self.client_instance_id = client_instance_id or str(uuid.uuid4())

    async def _connect(self) -> _ConnectionV2:
        validate_action_host_paths(self.socket_path, self.token_path)
        instance_token = load_instance_token(self.token_path)
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.setblocking(False)
        try:
            await asyncio.get_running_loop().sock_connect(sock, os.fspath(self.socket_path))
            assert_peer_uid(sock)
            challenge = HostChallengeV2.from_dict(
                decode_handshake_frame_v2(
                    await _read_frame(sock, MAX_HANDSHAKE_FRAME_BYTES)
                )
            )
            auth_key = self.auth_keys.load(challenge.key_version)
            response = ClientResponseV2.create(
                challenge=challenge,
                auth_key=auth_key,
                instance_token=instance_token,
                client_instance_id=self.client_instance_id,
            )
            await asyncio.get_running_loop().sock_sendall(
                sock, encode_frame(response.to_dict(), maximum=MAX_HANDSHAKE_FRAME_BYTES)
            )
            ready = HostReadyV2.from_dict(
                decode_handshake_frame_v2(
                    await _read_frame(sock, MAX_HANDSHAKE_FRAME_BYTES)
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
        except BaseException:
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
        except (KeyError, ValueError, ToolContractError) as exc:
            raise ActionHostProtocolError("host.protocol_mismatch") from exc
        return NativeExecutionResultV2(
            result_event_id=result["result_event_id"],
            request_id=result["request_id"],
            execution_id=result["execution_id"],
            tool_call_id=result["tool_call_id"],
            permit_id=result["permit_id"],
            idempotency_key=result["idempotency_key"],
            tool_id=result["tool_id"],
            tool_version=result["tool_version"],
            result_schema_sha256=result["result_schema_sha256"],
            registry_sha256=result["registry_sha256"],
            result_sha256=result["result_sha256"],
            outcome=outcome,
            result_payload=dict(result["result_payload"]),
            terminal_receipt=dict(receipt),
        )

    @asynccontextmanager
    async def _session(self):
        async with asyncio.timeout(self.timeout_seconds):
            connection = await self._connect()
            try:
                yield connection
            finally:
                connection.close()

    async def _receive_result(
        self, connection: _ConnectionV2, execution_id: str, idempotency_key: str,
    ) -> NativeExecutionResultV2:
        result = None
        terminal_receipt = None
        while result is None or terminal_receipt is None:
            message = await connection.receive()
            kind, payload = message["type"], message["payload"]
            if kind == "action.error":
                raise ActionHostProtocolError(payload["reason_code"])
            if kind in {"action.result", "action.result.snapshot"}:
                candidate = payload if kind == "action.result" else payload["result"]
                if result is not None and candidate != result:
                    raise ActionHostProtocolError("host.result_conflict")
                result = candidate
                records = [candidate]
            elif kind in {"action.receipt", "action.receipt.snapshot"}:
                records = [payload] if kind == "action.receipt" else [payload, *payload["receipts"]]
                for receipt in records:
                    if receipt.get("stage") == "VERIFICATION_COMPLETE":
                        if terminal_receipt is not None and receipt != terminal_receipt:
                            raise ActionHostProtocolError("host.result_conflict")
                        terminal_receipt = receipt
                if kind == "action.receipt.snapshot" and terminal_receipt is None:
                    raise ActionHostProtocolError("host.receipt_unknown")
            else:
                raise ActionHostProtocolError("host.protocol_mismatch")
            if any(record["execution_id"] != execution_id or record["idempotency_key"] != idempotency_key
                   for record in records):
                raise ActionHostProtocolError("host.result_conflict")
        return self._native_result(result, terminal_receipt)

    async def execute(self, execute_payload: Mapping[str, Any]) -> NativeExecutionResultV2:
        async with self._session() as connection:
            await connection.send({"type": "action.execute", "payload": dict(execute_payload)})
            result = await self._receive_result(
                connection, execute_payload["execution_id"], execute_payload["idempotency_key"]
            )
            if any(getattr(result, name) != execute_payload[name] for name in (
                "request_id", "tool_call_id", "permit_id", "tool_id", "tool_version", "registry_sha256"
            )):
                raise ActionHostProtocolError("host.result_conflict")
            return result

    async def query_result(self, execution_id: str, idempotency_key: str) -> NativeExecutionResultV2:
        async with self._session() as connection:
            query = {"execution_id": execution_id, "idempotency_key": idempotency_key}
            await connection.send({"type": "action.result.query", "payload": query})
            await connection.send({"type": "action.receipt.query", "payload": query})
            return await self._receive_result(connection, execution_id, idempotency_key)

    async def acknowledge(self, result: NativeExecutionResultV2) -> None:
        async with self._session() as connection:
            await connection.send({"type": "action.result.ack", "payload": {
                "result_event_id": result.result_event_id, "execution_id": result.execution_id,
                "idempotency_key": result.idempotency_key, "result_sha256": result.result_sha256,
            }})
            ping = {"nonce": base64.urlsafe_b64encode(os.urandom(16)).rstrip(b"=").decode("ascii")}
            await connection.send({"type": "action.ping", "payload": ping})
            if await connection.receive() != {"type": "action.pong", "payload": ping}:
                raise ActionHostProtocolError("host.result_unknown")
