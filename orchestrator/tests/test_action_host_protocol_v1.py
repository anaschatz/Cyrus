from __future__ import annotations

import hashlib
import hmac
import unittest
import uuid
from datetime import UTC, datetime, timedelta

import _bootstrap  # noqa: F401

from jarvis_orchestrator.action_host_protocol import (
    ACTION_HOST_PROTOCOL,
    ActionHostProtocolError,
    ClientResponseV1,
    Direction,
    HostChallengeV1,
    SessionKeysV1,
    decode_frame,
    encode_frame,
    hkdf_extract,
)
from jarvis_orchestrator.jcs import jcs_bytes


class ActionHostProtocolV1Tests(unittest.TestCase):
    def keys(self) -> SessionKeysV1:
        auth_key = bytes(range(32))
        instance_token = bytes(range(32, 64))
        k_boot = hkdf_extract(instance_token, auth_key)
        unsigned_challenge = HostChallengeV1(
            str(uuid.UUID("00000000-0000-4000-8000-000000000001")),
            bytes(range(16)),
            7,
            (datetime.now(UTC) + timedelta(minutes=1)).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
            hashlib.sha256(instance_token).hexdigest(),
            "",
        )
        challenge = HostChallengeV1(
            unsigned_challenge.host_instance_id,
            unsigned_challenge.host_nonce,
            unsigned_challenge.key_version,
            unsigned_challenge.challenge_expires_at_utc,
            unsigned_challenge.token_sha256,
            hmac.new(k_boot, jcs_bytes(unsigned_challenge.unsigned()), hashlib.sha256).hexdigest(),
        )
        unsigned_response = ClientResponseV1(
            challenge.host_instance_id,
            str(uuid.UUID("00000000-0000-4000-8000-000000000002")),
            challenge.host_nonce,
            bytes(range(16, 32)),
            "",
        )
        response = ClientResponseV1(
            unsigned_response.host_instance_id,
            unsigned_response.client_instance_id,
            unsigned_response.host_nonce,
            unsigned_response.client_nonce,
            hmac.new(k_boot, jcs_bytes(unsigned_response.unsigned()), hashlib.sha256).hexdigest(),
        )
        return SessionKeysV1.derive(
            auth_key=auth_key,
            instance_token=instance_token,
            challenge=challenge,
            response=response,
            session_id=str(uuid.UUID("00000000-0000-4000-8000-000000000003")),
        )

    def test_length_prefixed_framing_is_exact(self) -> None:
        frame = encode_frame({"a": 1}, maximum=32)
        self.assertEqual(frame[:4], b"\x00\x00\x00\x07")
        self.assertEqual(decode_frame(frame, maximum=32), {"a": 1})
        with self.assertRaises(ActionHostProtocolError):
            decode_frame(b"\x00\x00\x00\x00", maximum=32)

    def test_directional_keys_and_prefixes_are_distinct(self) -> None:
        keys = self.keys()
        self.assertNotEqual(keys.c2h_key, keys.h2c_key)
        self.assertNotEqual(keys.c2h_prefix, keys.h2c_prefix)

    def test_encrypted_execute_round_trip_and_replay_rejection(self) -> None:
        keys = self.keys()
        sender = keys.codec(Direction.C2H)
        receiver = keys.codec(Direction.C2H)
        payload = {"title": "note", "body": "private"}
        message = {
            "type": "action.execute",
            "payload": {
                "request_id": str(uuid.uuid4()),
                "execution_id": str(uuid.uuid4()),
                "tool_call_id": str(uuid.uuid4()),
                "permit_id": str(uuid.uuid4()),
                "idempotency_key": str(uuid.uuid4()),
                "tool_id": "note.capture@1",
                "tool_version": 1,
                "tool_schema_sha256": "a" * 64,
                "registry_sha256": "b" * 64,
                "policy_sha256": "c" * 64,
                "scheduler_contract_sha256": "d" * 64,
                "user_intent_binding_sha256": "e" * 64,
                "arguments_sha256": hashlib.sha256(jcs_bytes(payload)).hexdigest(),
                "payload": payload,
                "deadline_utc": "2026-08-11T16:00:00.000Z",
            },
        }
        frame = sender.encrypt(message)
        self.assertEqual(receiver.decrypt(frame), message)
        with self.assertRaisesRegex(ActionHostProtocolError, "host.replay"):
            receiver.decrypt(frame)

    def test_receipt_stage_is_closed(self) -> None:
        keys = self.keys()
        sender = keys.codec(Direction.H2C)
        receiver = keys.codec(Direction.H2C)
        identifiers = [str(uuid.uuid4()) for _ in range(3)]
        message = {
            "type": "action.receipt",
            "payload": {
                "request_id": identifiers[0],
                "execution_id": identifiers[1],
                "idempotency_key": identifiers[2],
                "stage": "HOST_RECEIVED",
                "observed_at_utc": "2026-08-11T15:00:00.000Z",
                "adapter_status": "PENDING",
                "os_receipt_sha256": "",
                "result_sha256": "",
                "verification": "NOT_APPLICABLE",
                "reason_code": "ok",
            },
        }
        self.assertEqual(receiver.decrypt(sender.encrypt(message)), message)
        message["payload"]["stage"] = "SCHEDULED"
        with self.assertRaises(ActionHostProtocolError):
            sender.encrypt(message)


if __name__ == "__main__":
    unittest.main()
