from __future__ import annotations

import hashlib
import struct
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
    MAX_ENCRYPTED_FRAME_BYTES,
    MAX_RESULT_PAYLOAD_BYTES,
    CLIENT_MESSAGE_TYPES_V2,
    SERVER_MESSAGE_TYPES_V2,
    ClientResponseV2,
    DirectionV2,
    HostChallengeV2,
    HostReadyV2,
    SessionKeysV2,
    decode_handshake_frame_v2,
    result_receipt_matches,
    validate_action_result_payload,
    validate_message_v2,
)
from jarvis_orchestrator.jcs import jcs_bytes
from jarvis_orchestrator.tool_contracts import ToolRegistryV1


def large_calendar_payload():
    event = dict(event_ref=str(uuid.uuid4()), calendar_label="Test", title="Test",
        start_local="2026-09-06T12:00:00.000", end_local="2026-09-06T13:00:00.000",
        start_utc="2026-09-06T09:00:00.000Z", end_utc="2026-09-06T10:00:00.000Z",
        timezone="Europe/Athens", all_day=False, location="", notes="")
    result = dict(observed_at_utc="2026-09-06T09:00:00.000Z",
                  events=[dict(event, event_ref=str(uuid.uuid4())) for _ in range(50)],
                  next_cursor="", truncated=False)
    remaining = MAX_RESULT_PAYLOAD_BYTES - len(jcs_bytes(result))
    for event in result["events"]:
        size = min(remaining, 8192)
        event["notes"] = "𐀀" * (size // 4) + ("", "a", "α", "€")[size % 4]
        remaining -= size
    assert remaining == 0 and len(jcs_bytes(result)) == MAX_RESULT_PAYLOAD_BYTES
    return result


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
            "6a2900bfef2168ffc76244b5a1fad910ff690a7c6505ff46aabf9dece1515f14",
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
        for change in ({"request_id": str(uuid.uuid4())}, {"verification": "MISMATCH"},
                       {"reason_code": "host.adapter_rejected"}, {"adapter_status": "REJECTED"}):
            with self.subTest(change=change):
                self.assertFalse(result_receipt_matches(result, receipt | change))
        receipt["result_sha256"] = "d" * 64
        self.assertFalse(result_receipt_matches(result, receipt))



    def test_malformed_wire_types_raise_protocol_errors(self):
        for value in ([], {}, None, True):
            with self.subTest(value=value), self.assertRaises(ActionHostProtocolError):
                validate_message_v2({"type": value, "payload": {}}, direction=DirectionV2.C2H)
            for name in ("status", "verification", "observed_at_utc"):
                with self.subTest(name=name, value=value), self.assertRaises(ActionHostProtocolError):
                    validate_action_result_payload(self.result() | {name: value})
        challenge = HostChallengeV2.create(auth_key=bytes(32), instance_token=bytes(32),
            host_instance_id=str(uuid.uuid4()), key_version=1).to_dict()
        for value in (1.0, "1", [], None, True):
            with self.subTest(key_version=value), self.assertRaises(ActionHostProtocolError):
                HostChallengeV2.from_dict(challenge | {"key_version": value})

    def test_deep_and_noncanonical_handshake_json_is_typed(self):
        for raw in (b'{"x":' + b'[' * 1500 + b'0' + b']' * 1500 + b'}',
                    b'{"x":1e999}', b'{"x":"\\ud800"}'):
            with self.subTest(raw=raw[:30]), self.assertRaises(ActionHostProtocolError):
                decode_handshake_frame_v2(struct.pack(">I", len(raw)) + raw)


    def test_maximum_schema_valid_result_fits_encrypted_frame(self):
        payload = large_calendar_payload()
        ToolRegistryV1.build(root_ids=("documents",), app_ids=("finder",)).validate_output(
            "calendar.read@1", payload)
        result = self.result() | {"result_payload": payload,
            "result_sha256": hashlib.sha256(jcs_bytes(payload)).hexdigest()}
        keys = self.handshake()
        sender, receiver = keys.codec(DirectionV2.H2C), keys.codec(DirectionV2.H2C)
        for message in ({"type": "action.result", "payload": result},
                        {"type": "action.result.snapshot", "payload": {"result": result}}):
            frame = sender.encrypt(message)
            self.assertLessEqual(len(frame) - 4, MAX_ENCRYPTED_FRAME_BYTES)
            self.assertEqual(receiver.decrypt(frame), message)
        too_large = result | {"result_payload": {"text": "x" * MAX_RESULT_PAYLOAD_BYTES}}
        with self.assertRaisesRegex(ActionHostProtocolError, "host.frame_too_large"):
            validate_action_result_payload(too_large)

if __name__ == "__main__":
    unittest.main()
