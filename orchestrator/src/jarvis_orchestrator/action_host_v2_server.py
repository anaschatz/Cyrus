"""Durable Action Host v2 server and closed native-adapter boundary."""

from __future__ import annotations

import hashlib
import json
import os
import socket
import sqlite3
import struct
import threading
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Protocol

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from .action_host_protocol import (
    ActionHostProtocolError,
    encode_frame,
    hkdf_extract,
)
from .action_host_security import assert_peer_uid, load_instance_token
from .action_host_v2_protocol import (
    MAX_ENCRYPTED_FRAME_BYTES,
    MAX_HANDSHAKE_FRAME_BYTES,
    ClientResponseV2,
    DirectionV2,
    HostChallengeV2,
    HostReadyV2,
    ResultVerificationV2,
    SessionKeysV2,
    decode_handshake_frame_v2,
)
from .domain import parse_utc
from .jcs import jcs_bytes, jcs_text
from .tool_contracts import ToolRegistryV1, ToolStatus


class NativeAdapterV2(Protocol):
    def execute(
        self,
        *,
        tool_id: str,
        arguments: Mapping[str, Any],
        execution_id: str,
        deadline_utc: str,
    ) -> "AdapterResultV2": ...


class HostAuthKeyProvider(Protocol):
    def active_version(self) -> int: ...
    def load(self, version: int) -> bytes: ...


@dataclass(frozen=True, slots=True)
class AdapterResultV2:
    status: ToolStatus
    verification: ResultVerificationV2
    reason_code: str
    result_payload: Mapping[str, Any]
    os_receipt_sha256: str = ""

    def __post_init__(self) -> None:
        allowed = {
            (ToolStatus.VERIFIED, ResultVerificationV2.EXACT_READBACK),
            (ToolStatus.ACCEPTED, ResultVerificationV2.UNAVAILABLE),
            (ToolStatus.FAILED, ResultVerificationV2.NOT_APPLICABLE),
            (ToolStatus.AMBIGUOUS, ResultVerificationV2.MISMATCH),
            (ToolStatus.AMBIGUOUS, ResultVerificationV2.INCONCLUSIVE),
        }
        if (self.status, self.verification) not in allowed:
            raise ValueError("invalid adapter result outcome")


def _utc_text(value: datetime | None = None) -> str:
    return (value or datetime.now(UTC)).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )


def _read_exact(connection: socket.socket, length: int) -> bytes:
    chunks: list[bytes] = []
    remaining = length
    while remaining:
        chunk = connection.recv(remaining)
        if not chunk:
            raise EOFError
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _read_frame(connection: socket.socket, maximum: int) -> bytes:
    prefix = _read_exact(connection, 4)
    length = struct.unpack(">I", prefix)[0]
    if length == 0 or length > maximum:
        raise ActionHostProtocolError("host.frame_too_large")
    return prefix + _read_exact(connection, length)


class ActionHostResultJournalV2:
    """Content-minimized receipts plus encrypted terminal result payloads."""

    def __init__(self, path: Path, auth_keys: HostAuthKeyProvider) -> None:
        self.path = Path(path)
        self.auth_keys = auth_keys
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.path.parent, 0o700)
        self._connection = sqlite3.connect(self.path, isolation_level=None, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA synchronous=FULL")
        self._connection.execute("PRAGMA foreign_keys=ON")
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS executions(
                execution_id TEXT PRIMARY KEY,
                idempotency_key TEXT NOT NULL UNIQUE,
                request_sha256 TEXT NOT NULL,
                request_id TEXT NOT NULL,
                tool_call_id TEXT NOT NULL,
                permit_id TEXT NOT NULL,
                tool_id TEXT NOT NULL,
                registry_sha256 TEXT NOT NULL,
                result_schema_sha256 TEXT NOT NULL,
                state TEXT NOT NULL,
                result_event_id TEXT,
                result_sha256 TEXT,
                key_version INTEGER,
                nonce BLOB,
                ciphertext_and_tag BLOB,
                aad_sha256 TEXT,
                acknowledged_at TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS receipts(
                execution_id TEXT NOT NULL REFERENCES executions(execution_id) ON DELETE CASCADE,
                stage TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                payload_sha256 TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY(execution_id,stage)
            );
            """
        )
        os.chmod(self.path, 0o600)
        self._lock = threading.RLock()
        self._recover_uncertain()

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def _transaction(self):
        class Transaction:
            def __init__(self, journal):
                self.journal = journal

            def __enter__(self):
                self.journal._lock.acquire()
                self.journal._connection.execute("BEGIN IMMEDIATE")
                return self.journal._connection

            def __exit__(self, exc_type, exc, tb):
                try:
                    self.journal._connection.execute("ROLLBACK" if exc_type else "COMMIT")
                finally:
                    self.journal._lock.release()
        return Transaction(self)

    def _recover_uncertain(self) -> None:
        with self._transaction() as connection:
            connection.execute(
                """UPDATE executions SET state='AMBIGUOUS',updated_at=?
                   WHERE state IN ('OS_CALL_ENTERING','OS_CALL_RETURNED')
                     AND result_sha256 IS NULL""",
                (_utc_text(),),
            )

    def admit(self, execute: Mapping[str, Any], *, result_schema_sha256: str) -> bool:
        request_digest = hashlib.sha256(jcs_bytes(execute)).hexdigest()
        stamp = _utc_text()
        with self._transaction() as connection:
            existing = connection.execute(
                "SELECT request_sha256,state FROM executions WHERE idempotency_key=?",
                (execute["idempotency_key"],),
            ).fetchone()
            if existing is not None:
                if existing["request_sha256"] != request_digest:
                    raise ActionHostProtocolError("host.idempotency_conflict")
                return False
            connection.execute(
                """INSERT INTO executions(
                   execution_id,idempotency_key,request_sha256,request_id,tool_call_id,
                   permit_id,tool_id,registry_sha256,result_schema_sha256,state,
                   created_at,updated_at
                   ) VALUES(?,?,?,?,?,?,?,?,?,'HOST_RECEIVED',?,?)""",
                (
                    execute["execution_id"],
                    execute["idempotency_key"],
                    request_digest,
                    execute["request_id"],
                    execute["tool_call_id"],
                    execute["permit_id"],
                    execute["tool_id"],
                    execute["registry_sha256"],
                    result_schema_sha256,
                    stamp,
                    stamp,
                ),
            )
        return True

    def store_receipt(self, execution_id: str, payload: Mapping[str, Any]) -> None:
        encoded = jcs_text(payload)
        digest = hashlib.sha256(encoded.encode()).hexdigest()
        with self._transaction() as connection:
            existing = connection.execute(
                "SELECT payload_sha256 FROM receipts WHERE execution_id=? AND stage=?",
                (execution_id, payload["stage"]),
            ).fetchone()
            if existing is not None:
                if existing["payload_sha256"] != digest:
                    raise ActionHostProtocolError("host.result_conflict")
                return
            connection.execute(
                "INSERT INTO receipts VALUES(?,?,?,?,?)",
                (execution_id, payload["stage"], encoded, digest, _utc_text()),
            )
            connection.execute(
                "UPDATE executions SET state=?,updated_at=? WHERE execution_id=?",
                (payload["stage"], _utc_text(), execution_id),
            )

    def store_result(self, result: Mapping[str, Any]) -> None:
        execution_id = result["execution_id"]
        result_bytes = jcs_bytes(result)
        result_sha = result["result_sha256"]
        key_version = self.auth_keys.active_version()
        key = self.auth_keys.load(key_version)
        if len(key) != 32:
            raise ActionHostProtocolError("host.auth_failed")
        nonce = os.urandom(12)
        aad = jcs_bytes(
            {
                "contract": "jarvis.action-host-result-at-rest/2",
                "execution_id": execution_id,
                "key_version": key_version,
                "result_event_id": result["result_event_id"],
                "result_sha256": result_sha,
                "tool_id": result["tool_id"],
            }
        )
        ciphertext = AESGCM(key).encrypt(nonce, result_bytes, aad)
        with self._transaction() as connection:
            row = connection.execute(
                "SELECT result_sha256 FROM executions WHERE execution_id=?",
                (execution_id,),
            ).fetchone()
            if row is None:
                raise ActionHostProtocolError("host.result_unknown")
            if row["result_sha256"] is not None:
                if row["result_sha256"] != result_sha:
                    raise ActionHostProtocolError("host.result_conflict")
                return
            connection.execute(
                """UPDATE executions SET state='RESULT_STORED',result_event_id=?,
                   result_sha256=?,key_version=?,nonce=?,ciphertext_and_tag=?,
                   aad_sha256=?,updated_at=? WHERE execution_id=?""",
                (
                    result["result_event_id"],
                    result_sha,
                    key_version,
                    nonce,
                    ciphertext,
                    hashlib.sha256(aad).hexdigest(),
                    _utc_text(),
                    execution_id,
                ),
            )

    def result(self, execution_id: str, idempotency_key: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM executions WHERE execution_id=? AND idempotency_key=?",
                (execution_id, idempotency_key),
            ).fetchone()
        if row is None or row["result_sha256"] is None or row["acknowledged_at"] is not None:
            return None
        key = self.auth_keys.load(row["key_version"])
        aad = jcs_bytes(
            {
                "contract": "jarvis.action-host-result-at-rest/2",
                "execution_id": execution_id,
                "key_version": row["key_version"],
                "result_event_id": row["result_event_id"],
                "result_sha256": row["result_sha256"],
                "tool_id": row["tool_id"],
            }
        )
        if hashlib.sha256(aad).hexdigest() != row["aad_sha256"]:
            raise ActionHostProtocolError("host.result_conflict")
        try:
            raw = AESGCM(key).decrypt(row["nonce"], row["ciphertext_and_tag"], aad)
        except Exception as exc:
            raise ActionHostProtocolError("host.result_conflict") from exc
        selected = json.loads(raw)
        if hashlib.sha256(jcs_bytes(selected["result_payload"])).hexdigest() != row["result_sha256"]:
            raise ActionHostProtocolError("host.result_conflict")
        return selected

    def receipts(self, execution_id: str, idempotency_key: str) -> list[dict[str, Any]]:
        with self._lock:
            exists = self._connection.execute(
                "SELECT 1 FROM executions WHERE execution_id=? AND idempotency_key=?",
                (execution_id, idempotency_key),
            ).fetchone()
            rows = self._connection.execute(
                """SELECT payload_json FROM receipts WHERE execution_id=?
                   ORDER BY CASE stage WHEN 'HOST_RECEIVED' THEN 1 WHEN 'OS_CALL_ENTERING' THEN 2
                   WHEN 'OS_CALL_RETURNED' THEN 3 ELSE 4 END""",
                (execution_id,),
            ).fetchall()
        if exists is None:
            raise ActionHostProtocolError("host.receipt_unknown")
        return [json.loads(row["payload_json"]) for row in rows]

    def acknowledge(
        self,
        *,
        result_event_id: str,
        execution_id: str,
        idempotency_key: str,
        result_sha256: str,
    ) -> None:
        with self._transaction() as connection:
            row = connection.execute(
                """SELECT result_event_id,result_sha256,acknowledged_at FROM executions
                   WHERE execution_id=? AND idempotency_key=?""",
                (execution_id, idempotency_key),
            ).fetchone()
            if row is None or row["result_event_id"] != result_event_id or row["result_sha256"] != result_sha256:
                raise ActionHostProtocolError("host.result_conflict")
            if row["acknowledged_at"] is None:
                connection.execute(
                    """UPDATE executions SET acknowledged_at=?,state='ACKNOWLEDGED',
                       nonce=NULL,ciphertext_and_tag=NULL,updated_at=? WHERE execution_id=?""",
                    (_utc_text(), _utc_text(), execution_id),
                )


class ActionHostV2Server:
    def __init__(
        self,
        *,
        socket_path: Path,
        token_path: Path,
        auth_keys: HostAuthKeyProvider,
        registry: ToolRegistryV1,
        journal: ActionHostResultJournalV2,
        adapter: NativeAdapterV2,
    ) -> None:
        self.socket_path = Path(socket_path)
        self.token_path = Path(token_path)
        self.auth_keys = auth_keys
        self.registry = registry
        self.journal = journal
        self.adapter = adapter
        self.host_instance_id = str(uuid.uuid4())

    def _receipt(
        self,
        execute: Mapping[str, Any],
        *,
        stage: str,
        adapter_status: str,
        result_sha256: str = "",
        verification: str = "NOT_APPLICABLE",
        reason_code: str = "ok",
        os_receipt_sha256: str = "",
    ) -> dict[str, Any]:
        return {
            "request_id": execute["request_id"],
            "execution_id": execute["execution_id"],
            "idempotency_key": execute["idempotency_key"],
            "stage": stage,
            "observed_at_utc": _utc_text(),
            "adapter_status": adapter_status,
            "os_receipt_sha256": os_receipt_sha256,
            "result_sha256": result_sha256,
            "verification": verification,
            "reason_code": reason_code,
        }

    @staticmethod
    def _receipt_verification(value: ResultVerificationV2) -> str:
        return {
            ResultVerificationV2.EXACT_READBACK: "MATCHED",
            ResultVerificationV2.UNAVAILABLE: "UNAVAILABLE",
            ResultVerificationV2.NOT_APPLICABLE: "NOT_APPLICABLE",
            ResultVerificationV2.MISMATCH: "MISMATCH",
            ResultVerificationV2.INCONCLUSIVE: "INCONCLUSIVE",
        }[value]

    def _send_error(self, sender, connection, execute, code: str) -> None:
        connection.sendall(
            sender.encrypt(
                {
                    "type": "action.error",
                    "payload": {
                        "request_id": execute.get("request_id", str(uuid.uuid4())),
                        "execution_id": execute.get("execution_id", str(uuid.uuid4())),
                        "reason_code": code,
                    },
                }
            )
        )

    def _execute(self, sender, connection, execute: Mapping[str, Any]) -> None:
        if parse_utc(execute["deadline_utc"]) <= datetime.now(UTC):
            self._send_error(sender, connection, execute, "host.deadline_expired")
            return
        definition = self.registry.resolve(execute["tool_id"])
        if (
            execute["registry_sha256"] != self.registry.digest
            or execute["tool_schema_sha256"] != definition.schema_sha256
        ):
            self._send_error(sender, connection, execute, "host.protocol_mismatch")
            return
        arguments = self.registry.validate(execute["tool_id"], execute["payload"])
        admitted = self.journal.admit(
            execute, result_schema_sha256=definition.output_schema_sha256
        )
        if not admitted:
            receipts = self.journal.receipts(execute["execution_id"], execute["idempotency_key"])
            result = self.journal.result(execute["execution_id"], execute["idempotency_key"])
            connection.sendall(
                sender.encrypt(
                    {
                        "type": "action.receipt.snapshot",
                        "payload": {
                            "execution_id": execute["execution_id"],
                            "idempotency_key": execute["idempotency_key"],
                            "receipts": receipts,
                        },
                    }
                )
            )
            if result is not None:
                connection.sendall(sender.encrypt({"type": "action.result.snapshot", "payload": {"result": result}}))
            return
        host_received = self._receipt(execute, stage="HOST_RECEIVED", adapter_status="PENDING")
        self.journal.store_receipt(execute["execution_id"], host_received)
        connection.sendall(sender.encrypt({"type": "action.receipt", "payload": host_received}))
        entering = self._receipt(execute, stage="OS_CALL_ENTERING", adapter_status="PENDING")
        self.journal.store_receipt(execute["execution_id"], entering)
        connection.sendall(sender.encrypt({"type": "action.receipt", "payload": entering}))
        try:
            adapter_result = self.adapter.execute(
                tool_id=execute["tool_id"],
                arguments=arguments,
                execution_id=execute["execution_id"],
                deadline_utc=execute["deadline_utc"],
            )
        except Exception:
            self._send_error(sender, connection, execute, "host.internal_safe_failure")
            return
        returned = self._receipt(
            execute,
            stage="OS_CALL_RETURNED",
            adapter_status=("REJECTED" if adapter_result.status is ToolStatus.FAILED else "RETURNED"),
            os_receipt_sha256=adapter_result.os_receipt_sha256,
            reason_code=adapter_result.reason_code,
        )
        self.journal.store_receipt(execute["execution_id"], returned)
        connection.sendall(sender.encrypt({"type": "action.receipt", "payload": returned}))
        if adapter_result.status not in {ToolStatus.VERIFIED, ToolStatus.ACCEPTED}:
            self._send_error(
                sender,
                connection,
                execute,
                "host.adapter_rejected" if adapter_result.status is ToolStatus.FAILED else "host.receipt_unknown",
            )
            return
        canonical_result = self.registry.validate_output(execute["tool_id"], adapter_result.result_payload)
        result_sha = hashlib.sha256(jcs_bytes(canonical_result)).hexdigest()
        result = {
            "result_event_id": str(uuid.uuid4()),
            "request_id": execute["request_id"],
            "execution_id": execute["execution_id"],
            "tool_call_id": execute["tool_call_id"],
            "permit_id": execute["permit_id"],
            "idempotency_key": execute["idempotency_key"],
            "tool_id": execute["tool_id"],
            "tool_version": execute["tool_version"],
            "result_schema_sha256": definition.output_schema_sha256,
            "registry_sha256": self.registry.digest,
            "status": adapter_result.status.value,
            "verification": adapter_result.verification.value,
            "reason_code": adapter_result.reason_code,
            "observed_at_utc": _utc_text(),
            "result_payload": canonical_result,
            "result_sha256": result_sha,
        }
        self.journal.store_result(result)
        verified = self._receipt(
            execute,
            stage="VERIFICATION_COMPLETE",
            adapter_status="RETURNED",
            result_sha256=result_sha,
            verification=self._receipt_verification(adapter_result.verification),
            reason_code=adapter_result.reason_code,
            os_receipt_sha256=adapter_result.os_receipt_sha256,
        )
        self.journal.store_receipt(execute["execution_id"], verified)
        connection.sendall(sender.encrypt({"type": "action.result", "payload": result}))
        connection.sendall(sender.encrypt({"type": "action.receipt", "payload": verified}))

    def handle_connection(self, connection: socket.socket) -> None:
        assert_peer_uid(connection)
        token = load_instance_token(self.token_path)
        key_version = self.auth_keys.active_version()
        auth_key = self.auth_keys.load(key_version)
        challenge = HostChallengeV2.create(
            auth_key=auth_key,
            instance_token=token,
            host_instance_id=self.host_instance_id,
            key_version=key_version,
        )
        connection.sendall(encode_frame(challenge.to_dict(), maximum=MAX_HANDSHAKE_FRAME_BYTES))
        response = ClientResponseV2.from_dict(
            decode_handshake_frame_v2(_read_frame(connection, MAX_HANDSHAKE_FRAME_BYTES))
        )
        session_id = str(uuid.uuid4())
        keys = SessionKeysV2.derive(
            auth_key=auth_key,
            instance_token=token,
            challenge=challenge,
            response=response,
            session_id=session_id,
        )
        ready = HostReadyV2.create(
            k_boot=hkdf_extract(token, auth_key),
            session_id=session_id,
            host_instance_id=self.host_instance_id,
        )
        connection.sendall(encode_frame(ready.to_dict(), maximum=MAX_HANDSHAKE_FRAME_BYTES))
        receiver = keys.codec(DirectionV2.C2H)
        sender = keys.codec(DirectionV2.H2C)
        while True:
            message = receiver.decrypt(_read_frame(connection, MAX_ENCRYPTED_FRAME_BYTES))
            if message["type"] == "action.execute":
                self._execute(sender, connection, message["payload"])
                return
            payload = message["payload"]
            if message["type"] == "action.result.query":
                result = self.journal.result(payload["execution_id"], payload["idempotency_key"])
                if result is None:
                    self._send_error(sender, connection, payload, "host.result_unknown")
                else:
                    connection.sendall(sender.encrypt({"type": "action.result.snapshot", "payload": {"result": result}}))
            elif message["type"] == "action.receipt.query":
                receipts = self.journal.receipts(payload["execution_id"], payload["idempotency_key"])
                connection.sendall(
                    sender.encrypt(
                        {
                            "type": "action.receipt.snapshot",
                            "payload": {
                                "execution_id": payload["execution_id"],
                                "idempotency_key": payload["idempotency_key"],
                                "receipts": receipts,
                            },
                        }
                    )
                )
            elif message["type"] == "action.result.ack":
                self.journal.acknowledge(**payload)
            elif message["type"] == "action.ping":
                connection.sendall(sender.encrypt({"type": "action.pong", "payload": payload}))
            elif message["type"] == "action.cancel":
                self._send_error(sender, connection, payload, "host.receipt_unknown")

    def serve_forever(self, stop_event: threading.Event) -> None:
        parent = self.socket_path.parent
        parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(parent, 0o700)
        if self.socket_path.exists():
            raise ActionHostProtocolError("host.auth_failed")
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            listener.bind(os.fspath(self.socket_path))
            os.chmod(self.socket_path, 0o600)
            listener.listen(8)
            listener.settimeout(0.25)
            while not stop_event.is_set():
                try:
                    connection, _ = listener.accept()
                except TimeoutError:
                    continue
                try:
                    self.handle_connection(connection)
                except (EOFError, OSError, ActionHostProtocolError):
                    pass
                finally:
                    connection.close()
        finally:
            listener.close()
            try:
                self.socket_path.unlink()
            except FileNotFoundError:
                pass
