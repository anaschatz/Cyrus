from __future__ import annotations

import hashlib
import unittest
import uuid

import _bootstrap  # noqa: F401

from jarvis_orchestrator.action_host_protocol import (
    ACTION_HOST_PROTOCOL,
    CLIENT_MESSAGE_TYPES,
    SERVER_MESSAGE_TYPES,
    ActionHostProtocolError,
    hkdf_extract,
)
from jarvis_orchestrator.action_host_v2_protocol import (
    ACTION_HOST_V2_CONTRACT_SHA256,
    ACTION_HOST_V2_PROTOCOL,
    CLIENT_MESSAGE_TYPES_V2,
    SERVER_MESSAGE_TYPES_V2,
    ClientResponseV2,
    DirectionV2,
    HostChallengeV2,
    HostReadyV2,
    SessionKeysV2,
    result_receipt_matches,
    validate_action_result_payload,
    validate_message_v2,
)
from jarvis_orchestrator.jcs import jcs_bytes


class ActionHostProtocolV2Tests(unittest.TestCase):
    def handshake(self):
        auth_key = bytes(range(32))
        token = bytes(reversed(range(32)))
        host_id = str(uuid.uuid4())
        client_id = str(uuid.uuid4())
        session_id = str(uuid.uuid4())
        challenge = HostChallengeV2.create(
            auth_key=auth_key,
            instance_token=token,
            host_instance_id=host_id,
            key_version=1,
        )
        response = ClientResponseV2.create(
            challenge=challenge,
            auth_key=auth_key,
            instance_token=token,
            client_instance_id=client_id,
        )
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
            host_instance_id=host_id,
        )
        ready.verify(hkdf_extract(token, auth_key))
        return keys

    def result(self):
        payload = {
            "events": [],
            "next_cursor": "",
            "observed_at_utc": "2026-08-11T18:00:00.000Z",
            "truncated": False,
        }
        return {
            "result_event_id": str(uuid.uuid4()),
            "request_id": str(uuid.uuid4()),
            "execution_id": str(uuid.uuid4()),
            "tool_call_id": str(uuid.uuid4()),
            "permit_id": str(uuid.uuid4()),
            "idempotency_key": str(uuid.uuid4()),
            "tool_id": "calendar.read@1",
            "tool_version": 1,
            "result_schema_sha256": "a" * 64,
            "registry_sha256": "b" * 64,
            "status": "VERIFIED",
            "verification": "EXACT_READBACK",
            "reason_code": "ok",
            "observed_at_utc": "2026-08-11T18:00:00.000Z",
            "result_payload": payload,
            "result_sha256": hashlib.sha256(jcs_bytes(payload)).hexdigest(),
        }

    def test_v2_is_additive_and_v1_message_sets_remain_receipt_only(self) -> None:
        self.assertEqual(ACTION_HOST_PROTOCOL, "jarvis.action-host/1")
        self.assertEqual(ACTION_HOST_V2_PROTOCOL, "jarvis.action-host/2")
        self.assertNotIn("action.result", SERVER_MESSAGE_TYPES)
        self.assertNotIn("action.result.query", CLIENT_MESSAGE_TYPES)
        self.assertIn("action.result", SERVER_MESSAGE_TYPES_V2)
        self.assertIn("action.result.query", CLIENT_MESSAGE_TYPES_V2)
        self.assertIn("action.result.ack", CLIENT_MESSAGE_TYPES_V2)

    def test_contract_digest_is_golden(self) -> None:
        self.assertEqual(
            ACTION_HOST_V2_CONTRACT_SHA256,
            "eab93d5724d271ea33623f098b2abb8f6be7bf00256e3925aa7b89745621c3c4",
        )

    def test_directional_v2_keys_are_distinct_and_round_trip_result(self) -> None:
        keys = self.handshake()
        self.assertNotEqual(keys.c2h_key, keys.h2c_key)
        self.assertNotEqual(keys.c2h_prefix, keys.h2c_prefix)
        sender = keys.codec(DirectionV2.H2C)
        receiver = keys.codec(DirectionV2.H2C)
        frame = sender.encrypt({"type": "action.result", "payload": self.result()})
        decoded = receiver.decrypt(frame)
        self.assertEqual(decoded["payload"]["status"], "VERIFIED")
        with self.assertRaisesRegex(ActionHostProtocolError, "host.replay"):
            receiver.decrypt(frame)

    def test_result_digest_unknown_fields_and_status_pair_fail_closed(self) -> None:
        result = self.result()
        validate_action_result_payload(result)
        tampered = dict(result)
        tampered["result_payload"] = {"events": [{"title": "changed"}]}
        with self.assertRaisesRegex(ActionHostProtocolError, "host.result_conflict"):
            validate_action_result_payload(tampered)
        unknown = dict(result)
        unknown["extra"] = True
        with self.assertRaisesRegex(ActionHostProtocolError, "host.protocol_mismatch"):
            validate_action_result_payload(unknown)
        invalid_pair = dict(result)
        invalid_pair["verification"] = "UNAVAILABLE"
        with self.assertRaisesRegex(ActionHostProtocolError, "host.protocol_mismatch"):
            validate_action_result_payload(invalid_pair)

    def test_result_ack_query_snapshot_and_matching_terminal_receipt(self) -> None:
        result = self.result()
        validate_message_v2(
            {
                "type": "action.result.query",
                "payload": {
                    "execution_id": result["execution_id"],
                    "idempotency_key": result["idempotency_key"],
                },
            },
            direction=DirectionV2.C2H,
        )
        validate_message_v2(
            {
                "type": "action.result.ack",
                "payload": {
                    "result_event_id": result["result_event_id"],
                    "execution_id": result["execution_id"],
                    "idempotency_key": result["idempotency_key"],
                    "result_sha256": result["result_sha256"],
                },
            },
            direction=DirectionV2.C2H,
        )
        validate_message_v2(
            {"type": "action.result.snapshot", "payload": {"result": result}},
            direction=DirectionV2.H2C,
        )
        receipt = {
            "request_id": result["request_id"],
            "execution_id": result["execution_id"],
            "idempotency_key": result["idempotency_key"],
            "stage": "VERIFICATION_COMPLETE",
            "observed_at_utc": "2026-08-11T18:00:00.000Z",
            "adapter_status": "RETURNED",
            "os_receipt_sha256": "c" * 64,
            "result_sha256": result["result_sha256"],
            "verification": "MATCHED",
            "reason_code": "ok",
        }
        self.assertTrue(result_receipt_matches(result, receipt))
        receipt["result_sha256"] = "d" * 64
        self.assertFalse(result_receipt_matches(result, receipt))


if __name__ == "__main__":
    unittest.main()
