"""Cloud transport rejects malformed frames before reserving a sequence."""

import json
import unittest
import uuid

from jarvis_voice.cloud_approval import (
    CLOUD_APPROVAL_PROTOCOL,
    MAX_EXTENSION_FRAME_BYTES,
    CloudApprovalEnvelope,
    CloudApprovalEventFactory,
)
from jarvis_voice.errors import ProtocolError

IDENTITY = str(uuid.uuid4())
STAMP = "2026-09-08T12:00:00.000Z"


def requested_payload():
    return {
        **dict.fromkeys(
            ("approval_id", "presentation_id", "task_id", "decision_nonce"), IDENTITY
        ),
        **dict.fromkeys(
            (
                "approval_binding_sha256",
                "egress_sha256",
                "route_plan_sha256",
                "stable_catalog_sha256",
                "qualification_sha256",
            ),
            "a" * 64,
        ),
        **dict.fromkeys(
            (
                "epoch",
                "billable_prompt_tokens",
                "input_token_limit",
                "output_token_limit",
                "max_reserved_microusd",
            ),
            1,
        ),
        "model_canonical_id": "test-model",
        "provider_endpoint_tag": "test-endpoint",
        "quantization": "none",
        "included_turn_ids": [],
        "expires_at": STAMP,
        "approve_key": "F10",
        "deny_key": "F11",
    }


def frame(event_type, payload):
    return dict(
        protocol=CLOUD_APPROVAL_PROTOCOL,
        type=event_type,
        event_id=IDENTITY,
        session_id=IDENTITY,
        seq=1,
        sent_at=STAMP,
        payload=payload,
    )


class CloudApprovalBoundaryTests(unittest.TestCase):
    def test_unhashable_wire_fields_raise_protocol_error(self):
        cases = [
            ([], {"protocol": CLOUD_APPROVAL_PROTOCOL}),
            ({}, {"protocol": CLOUD_APPROVAL_PROTOCOL}),
            ("cloud.approval.event_ack", {"acked_event_id": IDENTITY, "status": []}),
            (
                "cloud.approval.decision_ack",
                {"decision_event_id": IDENTITY, "status": {}},
            ),
            (
                "cloud.approval.decision",
                {
                    **dict.fromkeys(
                        (
                            "approval_id",
                            "presentation_id",
                            "requested_event_id",
                            "decision_nonce",
                        ),
                        IDENTITY,
                    ),
                    "approval_binding_sha256": "a" * 64,
                    "decision": [],
                },
            ),
            (
                "cloud.approval.invalidated",
                {
                    "approval_id": IDENTITY,
                    "presentation_id": IDENTITY,
                    "approval_binding_sha256": "a" * 64,
                    "reason_code": [],
                },
            ),
        ]
        for result, reason in (([], "approved"), ("applied", {})):
            cases.append(
                (
                    "cloud.approval.decision_result",
                    {
                        **dict.fromkeys(
                            ("decision_event_id", "approval_id", "presentation_id"),
                            IDENTITY,
                        ),
                        "approval_binding_sha256": "a" * 64,
                        "result": result,
                        "reason_code": reason,
                    },
                )
            )
        for event_type, payload in cases:
            with self.subTest(event_type=event_type, payload=payload):
                with self.assertRaises(ProtocolError):
                    CloudApprovalEnvelope.from_json(
                        json.dumps(frame(event_type, payload))
                    )

    def test_raw_and_escaped_surrogates_raise_protocol_error(self):
        payload = requested_payload()
        payload["model_canonical_id"] = "\ud800"
        for escaped in (False, True):
            with self.subTest(escaped=escaped), self.assertRaises(ProtocolError):
                CloudApprovalEnvelope.from_json(
                    json.dumps(
                        frame("cloud.approval.requested", payload), ensure_ascii=escaped
                    )
                )

    def test_rejected_creation_does_not_consume_sequence(self):
        for method in ("create", "reconstruct"):
            for bad_value in ("\ud800", "x" * MAX_EXTENSION_FRAME_BYTES):
                with self.subTest(method=method, value=repr(bad_value[:8])):
                    factory = CloudApprovalEventFactory(IDENTITY)
                    payload = requested_payload()
                    payload["model_canonical_id"] = bad_value
                    with self.assertRaises(ProtocolError):
                        getattr(factory, method)(
                            "cloud.approval.requested",
                            payload,
                            event_id=IDENTITY,
                            sent_at=STAMP,
                        )
                    event = factory.create(
                        "cloud.approval.ready", {"protocol": CLOUD_APPROVAL_PROTOCOL}
                    )
                    self.assertEqual(event.seq, 1)

    def test_invalid_payload_does_not_consume_sequence(self):
        factory = CloudApprovalEventFactory(IDENTITY)
        with self.assertRaises(ProtocolError):
            factory.create("cloud.approval.ready", {})
        self.assertEqual(
            factory.create(
                "cloud.approval.ready", {"protocol": CLOUD_APPROVAL_PROTOCOL}
            ).seq,
            1,
        )

    def test_utf8_byte_limit_allows_exact_boundary(self):
        payload = requested_payload()
        payload["model_canonical_id"] = ""
        base_size = len(
            json.dumps(
                frame("cloud.approval.requested", payload),
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        room = MAX_EXTENSION_FRAME_BYTES - base_size
        payload["model_canonical_id"] = "α" * (room // 2) + "x" * (room % 2)
        envelope = CloudApprovalEnvelope.from_dict(
            frame("cloud.approval.requested", payload)
        )
        wire = envelope.to_json().encode("utf-8")
        self.assertEqual(len(wire), MAX_EXTENSION_FRAME_BYTES)
        self.assertEqual(CloudApprovalEnvelope.from_json(wire), envelope)
        payload["model_canonical_id"] += "x"
        with self.assertRaises(ProtocolError):
            CloudApprovalEnvelope.from_dict(frame("cloud.approval.requested", payload))

    def test_deep_and_oversized_integer_json_raise_protocol_error(self):
        for wire in ("[" * 2000 + "0" + "]" * 2000, "1" * 5000):
            with self.subTest(prefix=wire[:8]), self.assertRaises(ProtocolError):
                CloudApprovalEnvelope.from_json(wire)
