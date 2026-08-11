"""Result-bearing ``jarvis.action-host/2`` authenticated wire contract.

Version 2 is deliberately additive.  The receipt-only v1 module is not
imported as a protocol authority and remains byte-for-byte frozen.  V2 adds a
single bounded terminal result, durable acknowledgement, and replay queries.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import os
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Any

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from .action_host_protocol import (
    ActionHostProtocolError,
    AdapterStatus,
    ReceiptStage,
    decode_frame,
    encode_frame,
    hkdf_expand,
    hkdf_extract,
    strict_json,
)
from .jcs import domain_sha256, jcs_bytes
from .tool_contracts import (
    ToolStatus,
    VerificationStatus,
    canonical_uuid4,
    parse_utc_strict,
)


ACTION_HOST_V2_PROTOCOL = "jarvis.action-host/2"
ACTION_HOST_V2_VERSION = "2"
MAX_HANDSHAKE_FRAME_BYTES = 8 * 1024
MAX_ENCRYPTED_FRAME_BYTES = 256 * 1024
MAX_RESULT_PAYLOAD_BYTES = 256 * 1024
NONCE_BYTES = 12


class DirectionV2(StrEnum):
    C2H = "C2H"
    H2C = "H2C"


class ResultVerificationV2(StrEnum):
    EXACT_READBACK = "EXACT_READBACK"
    UNAVAILABLE = "UNAVAILABLE"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    MISMATCH = "MISMATCH"
    INCONCLUSIVE = "INCONCLUSIVE"


CLIENT_MESSAGE_TYPES_V2 = frozenset(
    {
        "action.execute",
        "action.cancel",
        "action.receipt.query",
        "action.result.query",
        "action.result.ack",
        "action.ping",
    }
)
SERVER_MESSAGE_TYPES_V2 = frozenset(
    {
        "action.receipt",
        "action.receipt.snapshot",
        "action.result",
        "action.result.snapshot",
        "action.error",
        "action.pong",
    }
)
HOST_ERROR_CODES_V2 = frozenset(
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
        "host.result_unknown",
        "host.result_conflict",
        "host.internal_safe_failure",
    }
)

_RESULT_COMBINATIONS = frozenset(
    {
        (ToolStatus.VERIFIED.value, ResultVerificationV2.EXACT_READBACK.value),
        (ToolStatus.ACCEPTED.value, ResultVerificationV2.UNAVAILABLE.value),
        (ToolStatus.FAILED.value, ResultVerificationV2.NOT_APPLICABLE.value),
        (ToolStatus.AMBIGUOUS.value, ResultVerificationV2.MISMATCH.value),
        (ToolStatus.AMBIGUOUS.value, ResultVerificationV2.INCONCLUSIVE.value),
    }
)


def _b64(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _unb64(value: Any, expected: int | None = None) -> bytes:
    if not isinstance(value, str) or not value.isascii() or "=" in value:
        raise ActionHostProtocolError("host.auth_failed")
    try:
        raw = base64.b64decode(
            value + "=" * (-len(value) % 4), altchars=b"-_", validate=True
        )
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
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ActionHostProtocolError("host.protocol_mismatch")
    return value


def _exact(value: Any, fields: set[str]) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ActionHostProtocolError("host.protocol_mismatch")
    return dict(value)


def _utc_now_text() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def decode_handshake_frame_v2(frame: bytes) -> dict[str, Any]:
    row = decode_frame(frame, maximum=MAX_HANDSHAKE_FRAME_BYTES)
    if row.get("protocol") != ACTION_HOST_V2_PROTOCOL:
        raise ActionHostProtocolError("host.protocol_mismatch")
    return row


@dataclass(frozen=True, slots=True)
class HostChallengeV2:
    host_instance_id: str
    host_nonce: bytes
    key_version: int
    challenge_expires_at_utc: str
    token_sha256: str
    host_mac: str

    def unsigned(self) -> dict[str, Any]:
        if isinstance(self.key_version, bool) or self.key_version < 1:
            raise ActionHostProtocolError("host.protocol_mismatch")
        parse_utc_strict(self.challenge_expires_at_utc)
        return {
            "protocol": ACTION_HOST_V2_PROTOCOL,
            "type": "host.challenge",
            "supported_versions": [ACTION_HOST_V2_VERSION],
            "host_instance_id": _uuid(self.host_instance_id),
            "host_nonce": _b64(self.host_nonce),
            "key_version": self.key_version,
            "challenge_expires_at_utc": self.challenge_expires_at_utc,
            "token_sha256": _sha(self.token_sha256),
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self.unsigned(), "host_mac": _sha(self.host_mac)}

    @classmethod
    def create(
        cls,
        *,
        auth_key: bytes,
        instance_token: bytes,
        host_instance_id: str,
        key_version: int,
        ttl: timedelta = timedelta(seconds=10),
    ) -> "HostChallengeV2":
        if len(auth_key) != 32 or len(instance_token) != 32:
            raise ActionHostProtocolError("host.auth_failed")
        provisional = cls(
            _uuid(host_instance_id),
            os.urandom(16),
            key_version,
            (datetime.now(UTC) + ttl)
            .isoformat(timespec="milliseconds")
            .replace("+00:00", "Z"),
            hashlib.sha256(instance_token).hexdigest(),
            "0" * 64,
        )
        k_boot = hkdf_extract(instance_token, auth_key)
        mac = hmac.new(k_boot, jcs_bytes(provisional.unsigned()), hashlib.sha256).hexdigest()
        return cls(
            provisional.host_instance_id,
            provisional.host_nonce,
            provisional.key_version,
            provisional.challenge_expires_at_utc,
            provisional.token_sha256,
            mac,
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "HostChallengeV2":
        row = _exact(
            value,
            {
                "protocol",
                "type",
                "supported_versions",
                "host_instance_id",
                "host_nonce",
                "key_version",
                "challenge_expires_at_utc",
                "token_sha256",
                "host_mac",
            },
        )
        if (
            row["protocol"] != ACTION_HOST_V2_PROTOCOL
            or row["type"] != "host.challenge"
            or row["supported_versions"] != [ACTION_HOST_V2_VERSION]
        ):
            raise ActionHostProtocolError("host.protocol_mismatch")
        selected = cls(
            _uuid(row["host_instance_id"]),
            _unb64(row["host_nonce"], 16),
            row["key_version"],
            row["challenge_expires_at_utc"],
            _sha(row["token_sha256"]),
            _sha(row["host_mac"]),
        )
        selected.unsigned()
        return selected


@dataclass(frozen=True, slots=True)
class ClientResponseV2:
    host_instance_id: str
    client_instance_id: str
    host_nonce: bytes
    client_nonce: bytes
    client_mac: str

    def unsigned(self) -> dict[str, Any]:
        return {
            "protocol": ACTION_HOST_V2_PROTOCOL,
            "type": "client.response",
            "selected_version": ACTION_HOST_V2_VERSION,
            "host_instance_id": _uuid(self.host_instance_id),
            "client_instance_id": _uuid(self.client_instance_id),
            "host_nonce": _b64(self.host_nonce),
            "client_nonce": _b64(self.client_nonce),
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self.unsigned(), "client_mac": _sha(self.client_mac)}

    @classmethod
    def create(
        cls,
        *,
        challenge: HostChallengeV2,
        auth_key: bytes,
        instance_token: bytes,
        client_instance_id: str,
    ) -> "ClientResponseV2":
        k_boot = _verify_challenge(challenge, auth_key, instance_token)
        provisional = cls(
            challenge.host_instance_id,
            _uuid(client_instance_id),
            challenge.host_nonce,
            os.urandom(16),
            "0" * 64,
        )
        mac = hmac.new(k_boot, jcs_bytes(provisional.unsigned()), hashlib.sha256).hexdigest()
        return cls(
            provisional.host_instance_id,
            provisional.client_instance_id,
            provisional.host_nonce,
            provisional.client_nonce,
            mac,
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ClientResponseV2":
        row = _exact(
            value,
            {
                "protocol",
                "type",
                "selected_version",
                "host_instance_id",
                "client_instance_id",
                "host_nonce",
                "client_nonce",
                "client_mac",
            },
        )
        if (
            row["protocol"] != ACTION_HOST_V2_PROTOCOL
            or row["type"] != "client.response"
            or row["selected_version"] != ACTION_HOST_V2_VERSION
        ):
            raise ActionHostProtocolError("host.protocol_mismatch")
        selected = cls(
            _uuid(row["host_instance_id"]),
            _uuid(row["client_instance_id"]),
            _unb64(row["host_nonce"], 16),
            _unb64(row["client_nonce"], 16),
            _sha(row["client_mac"]),
        )
        selected.unsigned()
        return selected


def _verify_challenge(
    challenge: HostChallengeV2, auth_key: bytes, instance_token: bytes
) -> bytes:
    if len(auth_key) != 32 or len(instance_token) != 32:
        raise ActionHostProtocolError("host.auth_failed")
    if challenge.token_sha256 != hashlib.sha256(instance_token).hexdigest():
        raise ActionHostProtocolError("host.auth_failed")
    if parse_utc_strict(challenge.challenge_expires_at_utc).timestamp() <= time.time():
        raise ActionHostProtocolError("host.auth_failed")
    k_boot = hkdf_extract(instance_token, auth_key)
    expected = hmac.new(
        k_boot, jcs_bytes(challenge.unsigned()), hashlib.sha256
    ).hexdigest()
    if not hmac.compare_digest(expected, challenge.host_mac):
        raise ActionHostProtocolError("host.auth_failed")
    return k_boot


@dataclass(frozen=True, slots=True)
class HostReadyV2:
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
            "protocol": ACTION_HOST_V2_PROTOCOL,
            "type": "host.ready",
            "session_id": _uuid(self.session_id),
            "host_instance_id": _uuid(self.host_instance_id),
            "accepted": self.accepted,
            "reason_code": self.reason_code,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self.unsigned(), "host_mac": _sha(self.host_mac)}

    @classmethod
    def create(
        cls,
        *,
        k_boot: bytes,
        session_id: str,
        host_instance_id: str,
        accepted: bool = True,
        reason_code: str = "ok",
    ) -> "HostReadyV2":
        provisional = cls(session_id, host_instance_id, accepted, reason_code, "0" * 64)
        mac = hmac.new(k_boot, jcs_bytes(provisional.unsigned()), hashlib.sha256).hexdigest()
        return cls(session_id, host_instance_id, accepted, reason_code, mac)

    def verify(self, k_boot: bytes) -> None:
        expected = hmac.new(k_boot, jcs_bytes(self.unsigned()), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(expected, self.host_mac):
            raise ActionHostProtocolError("host.auth_failed")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "HostReadyV2":
        row = _exact(
            value,
            {
                "protocol",
                "type",
                "session_id",
                "host_instance_id",
                "accepted",
                "reason_code",
                "host_mac",
            },
        )
        if row["protocol"] != ACTION_HOST_V2_PROTOCOL or row["type"] != "host.ready":
            raise ActionHostProtocolError("host.protocol_mismatch")
        selected = cls(
            _uuid(row["session_id"]),
            _uuid(row["host_instance_id"]),
            row["accepted"],
            row["reason_code"],
            _sha(row["host_mac"]),
        )
        selected.unsigned()
        return selected


@dataclass(frozen=True, slots=True)
class SessionKeysV2:
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
        challenge: HostChallengeV2,
        response: ClientResponseV2,
        session_id: str,
    ) -> "SessionKeysV2":
        k_boot = _verify_challenge(challenge, auth_key, instance_token)
        expected_client = hmac.new(
            k_boot, jcs_bytes(response.unsigned()), hashlib.sha256
        ).hexdigest()
        if not hmac.compare_digest(expected_client, response.client_mac):
            raise ActionHostProtocolError("host.auth_failed")
        if (
            challenge.host_instance_id != response.host_instance_id
            or challenge.host_nonce != response.host_nonce
        ):
            raise ActionHostProtocolError("host.auth_failed")
        transcript = hashlib.sha256(
            jcs_bytes(challenge.unsigned()) + jcs_bytes(response.unsigned())
        ).digest()
        prk = hkdf_extract(transcript, k_boot)
        return cls(
            _uuid(session_id),
            hkdf_expand(prk, "jarvis.action-host/2|c2h|aes-256-gcm", 32),
            hkdf_expand(prk, "jarvis.action-host/2|h2c|aes-256-gcm", 32),
            hkdf_expand(prk, "jarvis.action-host/2|c2h|nonce-prefix", 32)[:4],
            hkdf_expand(prk, "jarvis.action-host/2|h2c|nonce-prefix", 32)[:4],
        )

    def codec(self, direction: DirectionV2) -> "EncryptedSessionCodecV2":
        if direction is DirectionV2.C2H:
            return EncryptedSessionCodecV2(
                self.session_id, direction, self.c2h_key, self.c2h_prefix
            )
        return EncryptedSessionCodecV2(
            self.session_id, direction, self.h2c_key, self.h2c_prefix
        )


class EncryptedSessionCodecV2:
    def __init__(
        self, session_id: str, direction: DirectionV2, key: bytes, prefix: bytes
    ) -> None:
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
            "protocol": ACTION_HOST_V2_PROTOCOL,
            "session_id": self.session_id,
            "direction": self.direction.value,
            "seq": sequence,
            "nonce": _b64(nonce),
        }
        validated = validate_message_v2(message, direction=self.direction)
        ciphertext = self._aead.encrypt(
            nonce, jcs_bytes(validated), jcs_bytes(header)
        )
        return encode_frame(
            {**header, "ciphertext_and_tag": _b64(ciphertext)},
            maximum=MAX_ENCRYPTED_FRAME_BYTES,
        )

    def decrypt(self, frame: bytes) -> dict[str, Any]:
        outer = decode_frame(frame, maximum=MAX_ENCRYPTED_FRAME_BYTES)
        row = _exact(
            outer,
            {"protocol", "session_id", "direction", "seq", "nonce", "ciphertext_and_tag"},
        )
        if (
            row["protocol"] != ACTION_HOST_V2_PROTOCOL
            or _uuid(row["session_id"]) != self.session_id
            or row["direction"] != self.direction.value
        ):
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
            plaintext = self._aead.decrypt(
                nonce, _unb64(row["ciphertext_and_tag"]), jcs_bytes(header)
            )
        except Exception as exc:
            raise ActionHostProtocolError("host.auth_failed") from exc
        message = strict_json(plaintext, maximum=MAX_ENCRYPTED_FRAME_BYTES)
        validated = validate_message_v2(message, direction=self.direction)
        self._receive_sequence = expected
        return validated


def _validate_execute(payload: Mapping[str, Any]) -> dict[str, Any]:
    row = _exact(
        payload,
        {
            "request_id",
            "execution_id",
            "tool_call_id",
            "permit_id",
            "idempotency_key",
            "tool_id",
            "tool_version",
            "tool_schema_sha256",
            "registry_sha256",
            "policy_sha256",
            "scheduler_contract_sha256",
            "user_intent_binding_sha256",
            "arguments_sha256",
            "payload",
            "deadline_utc",
        },
    )
    for name in ("request_id", "execution_id", "tool_call_id", "permit_id", "idempotency_key"):
        row[name] = _uuid(row[name])
    for name in (
        "tool_schema_sha256",
        "registry_sha256",
        "policy_sha256",
        "scheduler_contract_sha256",
        "user_intent_binding_sha256",
        "arguments_sha256",
    ):
        _sha(row[name])
    if (
        not isinstance(row["tool_id"], str)
        or not row["tool_id"].isascii()
        or isinstance(row["tool_version"], bool)
        or not isinstance(row["tool_version"], int)
        or row["tool_version"] < 1
    ):
        raise ActionHostProtocolError("host.protocol_mismatch")
    if (
        not isinstance(row["payload"], Mapping)
        or hashlib.sha256(jcs_bytes(row["payload"])).hexdigest()
        != row["arguments_sha256"]
    ):
        raise ActionHostProtocolError("host.idempotency_conflict")
    parse_utc_strict(row["deadline_utc"])
    return row


def _validate_receipt(payload: Mapping[str, Any]) -> dict[str, Any]:
    row = _exact(
        payload,
        {
            "request_id",
            "execution_id",
            "idempotency_key",
            "stage",
            "observed_at_utc",
            "adapter_status",
            "os_receipt_sha256",
            "result_sha256",
            "verification",
            "reason_code",
        },
    )
    for name in ("request_id", "execution_id", "idempotency_key"):
        row[name] = _uuid(row[name])
    try:
        ReceiptStage(row["stage"])
        AdapterStatus(row["adapter_status"])
        VerificationStatus(row["verification"])
    except (ValueError, TypeError) as exc:
        raise ActionHostProtocolError("host.protocol_mismatch") from exc
    parse_utc_strict(row["observed_at_utc"])
    _sha(row["os_receipt_sha256"], allow_empty=True)
    _sha(row["result_sha256"], allow_empty=True)
    if not isinstance(row["reason_code"], str):
        raise ActionHostProtocolError("host.protocol_mismatch")
    return row


def validate_action_result_payload(value: Mapping[str, Any]) -> dict[str, Any]:
    row = _exact(
        value,
        {
            "result_event_id",
            "request_id",
            "execution_id",
            "tool_call_id",
            "permit_id",
            "idempotency_key",
            "tool_id",
            "tool_version",
            "result_schema_sha256",
            "registry_sha256",
            "status",
            "verification",
            "reason_code",
            "observed_at_utc",
            "result_payload",
            "result_sha256",
        },
    )
    for name in (
        "result_event_id",
        "request_id",
        "execution_id",
        "tool_call_id",
        "permit_id",
        "idempotency_key",
    ):
        row[name] = _uuid(row[name])
    for name in ("result_schema_sha256", "registry_sha256", "result_sha256"):
        _sha(row[name])
    if (
        not isinstance(row["tool_id"], str)
        or not row["tool_id"].isascii()
        or isinstance(row["tool_version"], bool)
        or not isinstance(row["tool_version"], int)
        or row["tool_version"] < 1
        or (row["status"], row["verification"]) not in _RESULT_COMBINATIONS
        or not isinstance(row["reason_code"], str)
        or not row["reason_code"].isascii()
        or not 1 <= len(row["reason_code"]) <= 96
    ):
        raise ActionHostProtocolError("host.protocol_mismatch")
    parse_utc_strict(row["observed_at_utc"])
    if not isinstance(row["result_payload"], Mapping):
        raise ActionHostProtocolError("host.protocol_mismatch")
    encoded = jcs_bytes(row["result_payload"])
    if len(encoded) > MAX_RESULT_PAYLOAD_BYTES:
        raise ActionHostProtocolError("host.frame_too_large")
    if hashlib.sha256(encoded).hexdigest() != row["result_sha256"]:
        raise ActionHostProtocolError("host.result_conflict")
    return row


def validate_message_v2(
    value: Mapping[str, Any], *, direction: DirectionV2
) -> dict[str, Any]:
    row = _exact(value, {"type", "payload"})
    allowed = CLIENT_MESSAGE_TYPES_V2 if direction is DirectionV2.C2H else SERVER_MESSAGE_TYPES_V2
    if row["type"] not in allowed or not isinstance(row["payload"], Mapping):
        raise ActionHostProtocolError("host.protocol_mismatch")
    if row["type"] == "action.execute":
        row["payload"] = _validate_execute(row["payload"])
    elif row["type"] == "action.receipt":
        row["payload"] = _validate_receipt(row["payload"])
    elif row["type"] in {"action.cancel", "action.receipt.query", "action.result.query"}:
        payload = _exact(row["payload"], {"execution_id", "idempotency_key"})
        payload["execution_id"] = _uuid(payload["execution_id"])
        payload["idempotency_key"] = _uuid(payload["idempotency_key"])
        row["payload"] = payload
    elif row["type"] == "action.result.ack":
        payload = _exact(
            row["payload"],
            {"result_event_id", "execution_id", "idempotency_key", "result_sha256"},
        )
        for name in ("result_event_id", "execution_id", "idempotency_key"):
            payload[name] = _uuid(payload[name])
        _sha(payload["result_sha256"])
        row["payload"] = payload
    elif row["type"] == "action.result":
        row["payload"] = validate_action_result_payload(row["payload"])
    elif row["type"] == "action.result.snapshot":
        payload = _exact(row["payload"], {"result"})
        payload["result"] = validate_action_result_payload(payload["result"])
        row["payload"] = payload
    elif row["type"] == "action.receipt.snapshot":
        payload = _exact(row["payload"], {"execution_id", "idempotency_key", "receipts"})
        payload["execution_id"] = _uuid(payload["execution_id"])
        payload["idempotency_key"] = _uuid(payload["idempotency_key"])
        if not isinstance(payload["receipts"], list) or len(payload["receipts"]) > 4:
            raise ActionHostProtocolError("host.protocol_mismatch")
        receipts = [_validate_receipt(receipt) for receipt in payload["receipts"]]
        stages = [receipt["stage"] for receipt in receipts]
        expected = [stage.value for stage in ReceiptStage]
        if stages != expected[: len(stages)]:
            raise ActionHostProtocolError("host.protocol_mismatch")
        payload["receipts"] = receipts
        row["payload"] = payload
    elif row["type"] == "action.error":
        payload = _exact(row["payload"], {"request_id", "execution_id", "reason_code"})
        payload["request_id"] = _uuid(payload["request_id"])
        payload["execution_id"] = _uuid(payload["execution_id"])
        if payload["reason_code"] not in HOST_ERROR_CODES_V2:
            raise ActionHostProtocolError("host.protocol_mismatch")
        row["payload"] = payload
    elif row["type"] in {"action.ping", "action.pong"}:
        payload = _exact(row["payload"], {"nonce"})
        _unb64(payload["nonce"], 16)
        row["payload"] = payload
    return row


def result_receipt_matches(
    result: Mapping[str, Any], receipt: Mapping[str, Any]
) -> bool:
    selected_result = validate_action_result_payload(result)
    selected_receipt = _validate_receipt(receipt)
    return (
        selected_receipt["stage"] == ReceiptStage.VERIFICATION_COMPLETE.value
        and selected_receipt["execution_id"] == selected_result["execution_id"]
        and selected_receipt["idempotency_key"] == selected_result["idempotency_key"]
        and selected_receipt["result_sha256"] == selected_result["result_sha256"]
    )


def action_host_v2_contract_descriptor() -> dict[str, Any]:
    """Content-free descriptor used for qualification and golden digests."""

    return {
        "protocol": ACTION_HOST_V2_PROTOCOL,
        "version": ACTION_HOST_V2_VERSION,
        "c2h": sorted(CLIENT_MESSAGE_TYPES_V2),
        "h2c": sorted(SERVER_MESSAGE_TYPES_V2),
        "hkdf": [
            "jarvis.action-host/2|c2h|aes-256-gcm",
            "jarvis.action-host/2|h2c|aes-256-gcm",
            "jarvis.action-host/2|c2h|nonce-prefix",
            "jarvis.action-host/2|h2c|nonce-prefix",
        ],
        "maximum_handshake_frame_bytes": MAX_HANDSHAKE_FRAME_BYTES,
        "maximum_encrypted_frame_bytes": MAX_ENCRYPTED_FRAME_BYTES,
        "maximum_result_payload_bytes": MAX_RESULT_PAYLOAD_BYTES,
        "result_delivery": "one_terminal_result_durable_until_ack",
        "downgrade": "forbidden",
    }


ACTION_HOST_V2_CONTRACT_SHA256 = domain_sha256(
    "jarvis.action-host-contract/2", action_host_v2_contract_descriptor()
)
