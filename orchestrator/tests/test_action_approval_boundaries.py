from __future__ import annotations

import json
import unittest
import uuid

import _bootstrap  # noqa: F401
import test_action_approval_v1
from jarvis_voice.action_approval import ACTION_APPROVAL_PROTOCOL, ActionApprovalEnvelope, ActionApprovalEventFactory
from jarvis_voice.errors import ProtocolError


class ActionApprovalBoundaryTests(unittest.TestCase):
    def test_unhashable_wire_selectors_are_protocol_errors(self):
        base = test_action_approval_v1.ActionApprovalV1Tests().requested().to_dict()
        identity = {name: base["payload"][name] for name in ("approval_id", "presentation_id", "approval_binding_sha256")}
        cases = [
            ("action.approval.event_ack", {"acked_event_id": str(uuid.uuid4()), "status": "applied"}, "status"),
            ("action.approval.decision_ack", {"decision_event_id": str(uuid.uuid4()), "status": "accepted"}, "status"),
            ("action.approval.decision", dict(identity, requested_event_id=base["event_id"], decision_nonce=str(uuid.uuid4()), decision="deny"), "decision"),
            ("action.approval.decision_result", dict(identity, decision_event_id=str(uuid.uuid4()), result="rejected", reason_code="denied"), "result"),
            ("action.approval.decision_result", dict(identity, decision_event_id=str(uuid.uuid4()), result="rejected", reason_code="denied"), "reason_code"),
            ("action.approval.invalidated", dict(identity, reason_code="task_cancelled"), "reason_code"),
        ]
        for value in ([], {}):
            with self.subTest(field="type", value=value):
                with self.assertRaises(ProtocolError):
                    ActionApprovalEnvelope.from_json(json.dumps(dict(base, type=value)))
            for event_type, payload, field in cases:
                with self.subTest(event_type=event_type, field=field, value=value):
                    wire = json.dumps(dict(base, type=event_type, payload=dict(payload, **{field: value})))
                    with self.assertRaises(ProtocolError) as raised:
                        ActionApprovalEnvelope.from_json(wire)
                    self.assertEqual(raised.exception.code, "protocol.invalid_field")

    def test_malformed_json_is_bounded_and_typed(self):
        for raw in ("[" * 1200 + "]" * 1200, '{"number":' + "1" * 5000 + "}",
                    "\ud800", b"\xff", '{"a":1,"a":2}', "{", "NaN"):
            with self.subTest(raw=repr(raw[:20])):
                with self.assertRaises(ProtocolError) as raised:
                    ActionApprovalEnvelope.from_json(raw)
                self.assertEqual(raised.exception.code, "protocol.invalid_json")

    def test_rejected_factory_event_does_not_consume_a_sequence(self):
        factory = ActionApprovalEventFactory(str(uuid.uuid4()))
        for kwargs in ({"payload": {}}, {"payload": {"protocol": ACTION_APPROVAL_PROTOCOL}, "event_id": "invalid"},
                       {"payload": {"protocol": ACTION_APPROVAL_PROTOCOL}, "sent_at": "invalid"}):
            with self.subTest(kwargs=kwargs):
                before = factory.next_seq
                with self.assertRaises(ProtocolError):
                    factory.create("action.approval.ready", **kwargs)
                self.assertEqual(factory.next_seq, before)
                event = factory.create("action.approval.ready", {"protocol": ACTION_APPROVAL_PROTOCOL})
                self.assertEqual(event.seq, before)
                self.assertEqual(ActionApprovalEnvelope.from_json(event.to_json()), event)


if __name__ == "__main__":
    unittest.main()
