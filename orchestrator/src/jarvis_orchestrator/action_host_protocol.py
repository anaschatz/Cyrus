"""Frozen authenticated ``jarvis.action-host/1`` wire contract.

The codec is transport-only: it validates, authenticates, frames, and encrypts
messages but never performs a native action.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import struct
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDFExpand

from .jcs import jcs_bytes
from .tool_contracts import VerificationStatus, canonical_uuid4, parse_utc_strict


ACTION_HOST_PROTOCOL = "jarvis.action-host/1"
ACTION_HOST_VERSION = "1"
MAX_HANDSHAKE_FRAME_BYTES = 8 * 1024
MAX_ENCRYPTED_FRAME_BYTES = 256 * 1024
NONCE_BYTES = 12


class ActionHostProtocolError(ValueError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class Direction(StrEnum):
    C2H = "C2H"
    H2C = "H2C"


class ReceiptStage(StrEnum):
    HOST_RECEIVED = "HOST_RECEIVED"
    OS_CALL_ENTERING = "OS_CALL_ENTERING"
    OS_CALL_RETURNED = "OS_CALL_RETURNED"
    VERIFICATION_COMPLETE = "VERIFICATION_COMPLETE"


class AdapterStatus(StrEnum):
    PENDING = "PENDING"
    ACCEPTED = "ACCEPTED"
    REJECTED = "REJECTED"
    RETURNED = "RETURNED"


CLIENT_MESSAGE_TYPES = frozenset({"action.execute", "action.cancel", "action.receipt.query", "action.ping"})
SERVER_MESSAGE_TYPES = frozenset({"action.receipt", "action.receipt.snapshot", "action.error", "action.pong"})
HOST_ERROR_CODES = frozenset(
    {
        "host.protocol_mismatch",
        "host.auth_failed",
        "host.replay",
        "host.sequence_gap",
        "host.frame_too_large",
        "host.deadline_expired",
        "host.idempotency_conflict",
        "host.adapter_rejected",
        "host.receipt_unknown",
        "host.internal_safe_failure",
    }
)


def _b64(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _unb64(value: Any, expected: int | None = None) -> bytes:
    if not isinstance(value, str) or not value.isascii() or "=" in value:
        raise ActionHostProtocolError("host.auth_failed")
    try:
        raw = base64.b64decode(value + "=" * (-len(value) % 4), altchars=b"-_", validate=True)
    except (ValueError, base64.binascii.Error) as exc:
        raise ActionHostProtocolError("host.auth_failed") from exc
    if expected is not None and len(raw) != expected:
        raise ActionHostProtocolError("host.auth_failed")
    return raw


def _uuid(value: Any) -> str:
    try:
        return canonical_uuid4(value)
    except Exception as exc:
        raise ActionHostProtocolError("host.protocol_mismatch") from exc


def _sha(value: Any, *, allow_empty: bool = False) -> str:
    if allow_empty and value == "":
        return ""
    if not isinstance(value, str) or len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ActionHostProtocolError("host.protocol_mismatch")
    return value


def _exact(value: Any, fields: set[str]) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ActionHostProtocolError("host.protocol_mismatch")
    return dict(value)


def strict_json(raw: bytes, *, maximum: int) -> dict[str, Any]:
    if not raw or len(raw) > maximum:
        raise ActionHostProtocolError("host.frame_too_large")
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise ActionHostProtocolError("host.protocol_mismatch") from exc
    if text.startswith("\ufeff"):
        raise ActionHostProtocolError("host.protocol_mismatch")

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ActionHostProtocolError("host.protocol_mismatch")
            result[key] = value
        return result

    try:
        value = json.loads(
            text,
            object_pairs_hook=pairs,
            parse_constant=lambda _: (_ for _ in ()).throw(ActionHostProtocolError("host.protocol_mismatch")),
        )
    except ActionHostProtocolError:
        raise
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise ActionHostProtocolError("host.protocol_mismatch") from exc
    if not isinstance(value, dict) or jcs_bytes(value) != raw:
        raise ActionHostProtocolError("host.protocol_mismatch")
    return value


def encode_frame(value: Mapping[str, Any], *, maximum: int) -> bytes:
    body = jcs_bytes(value)
    if not body or len(body) > maximum:
        raise ActionHostProtocolError("host.frame_too_large")
    return struct.pack(">I", len(body)) + body


def decode_frame(frame: bytes, *, maximum: int) -> dict[str, Any]:
    if len(frame) < 4:
        raise ActionHostProtocolError("host.protocol_mismatch")
    length = struct.unpack(">I", frame[:4])[0]
    if length == 0 or length > maximum or len(frame) != length + 4:
        raise ActionHostProtocolError("host.frame_too_large")
    return strict_json(frame[4:], maximum=maximum)


def hkdf_extract(salt: bytes, ikm: bytes) -> bytes:
    return hmac.new(salt, ikm, hashlib.sha256).digest()


def hkdf_expand(prk: bytes, info: str, length: int) -> bytes:
    return HKDFExpand(algorithm=hashes.SHA256(), length=length, info=info.encode("ascii")).derive(prk)


@dataclass(frozen=True, slots=True)
class HostChallengeV1:
    host_instance_id: str
    host_nonce: bytes
    key_version: int
    challenge_expires_at_utc: str
    token_sha256: str
    host_mac: str

    def unsigned(self) -> dict[str, Any]:
        return {
            "protocol": ACTION_HOST_PROTOCOL,
            "type": "host.challenge",
            "supported_versions": [ACTION_HOST_VERSION],
            "host_instance_id": _uuid(self.host_instance_id),
            "host_nonce": _b64(self.host_nonce),
            "key_version": self.key_version,
            "challenge_expires_at_utc": self.challenge_expires_at_utc,
            "token_sha256": _sha(self.token_sha256),
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self.unsigned(), "host_mac": self.host_mac}


@dataclass(frozen=True, slots=True)
class ClientResponseV1:
    host_instance_id: str
    client_instance_id: str
    host_nonce: bytes
    client_nonce: bytes
    client_mac: str

    def unsigned(self) -> dict[str, Any]:
        return {
            "protocol": ACTION_HOST_PROTOCOL,
            "type": "client.response",
            "selected_version": ACTION_HOST_VERSION,
            "host_instance_id": _uuid(self.host_instance_id),
            "client_instance_id": _uuid(self.client_instance_id),
            "host_nonce": _b64(self.host_nonce),
            "client_nonce": _b64(self.client_nonce),
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self.unsigned(), "client_mac": self.client_mac}


@dataclass(frozen=True, slots=True)
class HostReadyV1:
    session_id: str
    host_instance_id: str
    accepted: bool
    reason_code: str
    host_mac: str

    def unsigned(self) -> dict[str, Any]:
        if not isinstance(self.accepted, bool):
            raise ActionHostProtocolError("host.protocol_mismatch")
        if self.reason_code not in {"ok", "host.protocol_mismatch", "host.auth_failed"}:
            raise ActionHostProtocolError("host.protocol_mismatch")
        return {
            "protocol": ACTION_HOST_PROTOCOL,
            "type": "host.ready",
            "session_id": _uuid(self.session_id),
            "host_instance_id": _uuid(self.host_instance_id),
            "accepted": self.accepted,
            "reason_code": self.reason_code,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self.unsigned(), "host_mac": _sha(self.host_mac)}

    def verify(self, k_boot: bytes) -> None:
        if len(k_boot) != 32:
            raise ActionHostProtocolError("host.auth_failed")
        expected = hmac.new(k_boot, jcs_bytes(self.unsigned()), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(expected, self.host_mac):
            raise ActionHostProtocolError("host.auth_failed")


@dataclass(frozen=True, slots=True)
class SessionKeysV1:
    session_id: str
    c2h_key: bytes
    h2c_key: bytes
    c2h_prefix: bytes
    h2c_prefix: bytes

    @classmethod
    def derive(
        cls,
        *,
        auth_key: bytes,
        instance_token: bytes,
        challenge: HostChallengeV1,
        response: ClientResponseV1,
        session_id: str,
    ) -> "SessionKeysV1":
        if len(auth_key) != 32 or len(instance_token) != 32:
            raise ActionHostProtocolError("host.auth_failed")
        k_boot = hkdf_extract(instance_token, auth_key)
        expected_host = hmac.new(k_boot, jcs_bytes(challenge.unsigned()), hashlib.sha256).hexdigest()
        expected_client = hmac.new(k_boot, jcs_bytes(response.unsigned()), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(expected_host, challenge.host_mac) or not hmac.compare_digest(expected_client, response.client_mac):
            raise ActionHostProtocolError("host.auth_failed")
        if challenge.host_instance_id != response.host_instance_id or challenge.host_nonce != response.host_nonce:
            raise ActionHostProtocolError("host.auth_failed")
        try:
            if parse_utc_strict(challenge.challenge_expires_at_utc).timestamp() <= time.time():
                raise ActionHostProtocolError("host.auth_failed")
        except Exception as exc:
            if isinstance(exc, ActionHostProtocolError):
                raise
            raise ActionHostProtocolError("host.auth_failed") from exc
        transcript = hashlib.sha256(jcs_bytes(challenge.unsigned()) + jcs_bytes(response.unsigned())).digest()
        prk = hkdf_extract(transcript, k_boot)
        return cls(
            _uuid(session_id),
            hkdf_expand(prk, "jarvis.action-host/1|c2h|aes-256-gcm", 32),
            hkdf_expand(prk, "jarvis.action-host/1|h2c|aes-256-gcm", 32),
            hkdf_expand(prk, "jarvis.action-host/1|c2h|nonce-prefix", 32)[:4],
            hkdf_expand(prk, "jarvis.action-host/1|h2c|nonce-prefix", 32)[:4],
        )

    def codec(self, direction: Direction) -> "EncryptedSessionCodecV1":
        if direction is Direction.C2H:
            return EncryptedSessionCodecV1(self.session_id, direction, self.c2h_key, self.c2h_prefix)
        return EncryptedSessionCodecV1(self.session_id, direction, self.h2c_key, self.h2c_prefix)


class EncryptedSessionCodecV1:
    def __init__(self, session_id: str, direction: Direction, key: bytes, prefix: bytes) -> None:
        if len(key) != 32 or len(prefix) != 4:
            raise ActionHostProtocolError("host.auth_failed")
        self.session_id = _uuid(session_id)
        self.direction = direction
        self._aead = AESGCM(key)
        self._prefix = bytes(prefix)
        self._send_sequence = 0
        self._receive_sequence = 0

    def _nonce(self, sequence: int) -> bytes:
        if not 1 <= sequence <= 2**64 - 1:
            raise ActionHostProtocolError("host.sequence_gap")
        return self._prefix + sequence.to_bytes(8, "big")

    def encrypt(self, message: Mapping[str, Any]) -> bytes:
        self._send_sequence += 1
        sequence = self._send_sequence
        nonce = self._nonce(sequence)
        header = {
            "protocol": ACTION_HOST_PROTOCOL,
            "session_id": self.session_id,
            "direction": self.direction.value,
            "seq": sequence,
            "nonce": _b64(nonce),
        }
        ciphertext = self._aead.encrypt(nonce, jcs_bytes(validate_message(message, direction=self.direction)), jcs_bytes(header))
        return encode_frame({**header, "ciphertext_and_tag": _b64(ciphertext)}, maximum=MAX_ENCRYPTED_FRAME_BYTES)

    def decrypt(self, frame: bytes) -> dict[str, Any]:
        outer = decode_frame(frame, maximum=MAX_ENCRYPTED_FRAME_BYTES)
        row = _exact(outer, {"protocol", "session_id", "direction", "seq", "nonce", "ciphertext_and_tag"})
        if row["protocol"] != ACTION_HOST_PROTOCOL or _uuid(row["session_id"]) != self.session_id or row["direction"] != self.direction.value:
            raise ActionHostProtocolError("host.protocol_mismatch")
        if isinstance(row["seq"], bool) or not isinstance(row["seq"], int):
            raise ActionHostProtocolError("host.sequence_gap")
        expected = self._receive_sequence + 1
        if row["seq"] < expected:
            raise ActionHostProtocolError("host.replay")
        if row["seq"] != expected:
            raise ActionHostProtocolError("host.sequence_gap")
        nonce = _unb64(row["nonce"], NONCE_BYTES)
        if nonce != self._nonce(expected):
            raise ActionHostProtocolError("host.replay")
        header = {name: row[name] for name in ("protocol", "session_id", "direction", "seq", "nonce")}
        try:
            plaintext = self._aead.decrypt(nonce, _unb64(row["ciphertext_and_tag"]), jcs_bytes(header))
        except Exception as exc:
            raise ActionHostProtocolError("host.auth_failed") from exc
        message = strict_json(plaintext, maximum=MAX_ENCRYPTED_FRAME_BYTES)
        validate_message(message, direction=self.direction)
        self._receive_sequence = expected
        return message


def _validate_execute(payload: Mapping[str, Any]) -> dict[str, Any]:
    row = _exact(payload, {"request_id", "execution_id", "tool_call_id", "permit_id", "idempotency_key", "tool_id", "tool_version", "tool_schema_sha256", "registry_sha256", "policy_sha256", "scheduler_contract_sha256", "user_intent_binding_sha256", "arguments_sha256", "payload", "deadline_utc"})
    for name in ("request_id", "execution_id", "tool_call_id", "permit_id", "idempotency_key"):
        row[name] = _uuid(row[name])
    for name in ("tool_schema_sha256", "registry_sha256", "policy_sha256", "scheduler_contract_sha256", "user_intent_binding_sha256", "arguments_sha256"):
        _sha(row[name])
    if not isinstance(row["tool_id"], str) or not row["tool_id"].isascii() or isinstance(row["tool_version"], bool) or not isinstance(row["tool_version"], int) or row["tool_version"] < 1:
        raise ActionHostProtocolError("host.protocol_mismatch")
    if not isinstance(row["payload"], Mapping) or hashlib.sha256(jcs_bytes(row["payload"])).hexdigest() != row["arguments_sha256"]:
        raise ActionHostProtocolError("host.idempotency_conflict")
    parse_utc_strict(row["deadline_utc"])
    return row


def _validate_receipt(payload: Mapping[str, Any]) -> dict[str, Any]:
    row = _exact(payload, {"request_id", "execution_id", "idempotency_key", "stage", "observed_at_utc", "adapter_status", "os_receipt_sha256", "result_sha256", "verification", "reason_code"})
    for name in ("request_id", "execution_id", "idempotency_key"):
        row[name] = _uuid(row[name])
    try:
        ReceiptStage(row["stage"])
        AdapterStatus(row["adapter_status"])
        VerificationStatus(row["verification"])
    except ValueError as exc:
        raise ActionHostProtocolError("host.protocol_mismatch") from exc
    parse_utc_strict(row["observed_at_utc"])
    _sha(row["os_receipt_sha256"], allow_empty=True)
    _sha(row["result_sha256"], allow_empty=True)
    if not isinstance(row["reason_code"], str) or row["reason_code"] not in HOST_ERROR_CODES | {"ok", "calendar.write_only_no_readback"}:
        raise ActionHostProtocolError("host.protocol_mismatch")
    return row


def validate_message(value: Mapping[str, Any], *, direction: Direction) -> dict[str, Any]:
    row = _exact(value, {"type", "payload"})
    allowed = CLIENT_MESSAGE_TYPES if direction is Direction.C2H else SERVER_MESSAGE_TYPES
    if row["type"] not in allowed or not isinstance(row["payload"], Mapping):
        raise ActionHostProtocolError("host.protocol_mismatch")
    if row["type"] == "action.execute":
        _validate_execute(row["payload"])
    elif row["type"] == "action.receipt":
        _validate_receipt(row["payload"])
    elif row["type"] in {"action.cancel", "action.receipt.query"}:
        payload = _exact(row["payload"], {"execution_id", "idempotency_key"})
        _uuid(payload["execution_id"])
        _uuid(payload["idempotency_key"])
    elif row["type"] == "action.error":
        payload = _exact(row["payload"], {"request_id", "execution_id", "reason_code"})
        _uuid(payload["request_id"])
        _uuid(payload["execution_id"])
        if payload["reason_code"] not in HOST_ERROR_CODES:
            raise ActionHostProtocolError("host.protocol_mismatch")
    elif row["type"] == "action.receipt.snapshot":
        payload = _exact(row["payload"], {"execution_id", "idempotency_key", "receipts"})
        _uuid(payload["execution_id"])
        _uuid(payload["idempotency_key"])
        if not isinstance(payload["receipts"], list) or len(payload["receipts"]) > 4:
            raise ActionHostProtocolError("host.protocol_mismatch")
        stages: list[str] = []
        for receipt in payload["receipts"]:
            stages.append(_validate_receipt(receipt)["stage"])
        expected_order = [stage.value for stage in ReceiptStage]
        if stages != expected_order[: len(stages)]:
            raise ActionHostProtocolError("host.protocol_mismatch")
    elif row["type"] in {"action.ping", "action.pong"}:
        _exact(row["payload"], {"nonce"})
        _unb64(row["payload"]["nonce"], 16)
    return row
