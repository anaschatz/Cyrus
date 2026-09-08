from __future__ import annotations

import json
import unittest

import _bootstrap  # noqa: F401

from jarvis_voice.action_approval import ActionApprovalEnvelope
from jarvis_voice.cloud_approval import CloudApprovalEnvelope
from jarvis_voice.errors import ProtocolError
from jarvis_voice.protocol import Envelope, EventFactory, new_uuid4
from jarvis_voice.scheduler_protocol import SchedulerEnvelope
from jarvis_voice.sequencing import SequenceResult, SequenceTracker

from helpers import speech_commit


class EnvelopeTests(unittest.TestCase):
    def test_payloads_must_be_serializable_as_finite_utf8_json(self) -> None:
        for bad in ("\ud800", {"nested": "\udfff"}, float("nan"), float("inf"), -float("inf")):
            with self.subTest(value=repr(bad)), self.assertRaisesRegex(ProtocolError, "protocol.invalid_json"):
                EventFactory().create("client.hello", {"extension": bad})
        event = EventFactory().create("response.text.delta", {
            "chunk_index": 0, "text": "Valid text."
        }, response_id=new_uuid4()).to_dict()
        event["payload"]["text"] = "\ud800"
        with self.assertRaisesRegex(ProtocolError, "protocol.invalid_json"):
            Envelope.from_json(json.dumps(event))

    def test_factory_and_ack_use_the_normalized_uuid_identity(self) -> None:
        session_id = new_uuid4()
        event_id = new_uuid4()
        factory = EventFactory(session_id=session_id.upper())
        self.assertEqual(factory.session_id, session_id)
        event = factory.create("server.ack", {"acked_event_id": event_id.upper(), "status": "accepted"})
        self.assertEqual(event.payload["acked_event_id"], event_id)

    def test_invalid_discriminator_types_have_protocol_errors(self) -> None:
        valid = EventFactory().create("server.ready", {"protocol": "jarvis.voice/1"}).to_dict()
        for field in ("v", "type"):
            with self.subTest(field=field), self.assertRaises(ProtocolError) as raised:
                Envelope.from_dict({**valid, field: {"private-frame-content": "secret"}})
            self.assertNotIn("private-frame-content", json.dumps(raised.exception.to_log_record()))
        for field, value in (("v", True), ("v", 1.0), ("type", [])):
            with self.subTest(field=field, value=value), self.assertRaises(ProtocolError):
                Envelope.from_dict({**valid, field: value})
        for event_type, payload in (
            ("server.ack", {"acked_event_id": new_uuid4(), "status": []}),
            ("response.cancelled", {"reason_code": {}, "partial_output": False}),
        ):
            with self.subTest(event_type=event_type), self.assertRaises(ProtocolError):
                EventFactory().create(event_type, payload, response_id=new_uuid4())

    def setUp(self) -> None:
        self.factory = EventFactory()

    def test_round_trip_utf8_and_unknown_optional_field(self) -> None:
        event = self.factory.create("server.ready", {"protocol": "jarvis.voice/1"})
        data = event.to_dict()
        data["future_optional"] = {"ignored": True}
        decoded = Envelope.from_json(json.dumps(data, ensure_ascii=False))
        self.assertEqual(decoded, event)
        self.assertIn("jarvis.voice/1", decoded.to_json())
        self.assertNotIn("future_optional", decoded.to_dict())

    def test_unknown_event_and_missing_field_are_typed_errors(self) -> None:
        event = self.factory.create("server.ready", {"protocol": "jarvis.voice/1"})
        data = event.to_dict()
        data["type"] = "future.unknown"
        with self.assertRaisesRegex(ProtocolError, "protocol.unknown_event"):
            Envelope.from_dict(data)
        data = event.to_dict()
        del data["event_id"]
        with self.assertRaisesRegex(ProtocolError, "protocol.missing_field"):
            Envelope.from_dict(data)

    def test_deeply_nested_json_is_a_typed_protocol_error(self) -> None:
        for envelope in (Envelope, ActionApprovalEnvelope, CloudApprovalEnvelope, SchedulerEnvelope):
            with self.subTest(envelope=envelope.__name__):
                with self.assertRaisesRegex(ProtocolError, "protocol.invalid_json"):
                    envelope.from_json('[' * 2000 + '0' + ']' * 2000)

    def test_response_and_playback_events_require_response_id(self) -> None:
        with self.assertRaisesRegex(ProtocolError, "protocol.missing_field"):
            self.factory.create("response.text.done", {})
        with self.assertRaisesRegex(ProtocolError, "protocol.missing_field"):
            self.factory.create("playback.completed", {})

    def test_greek_speech_text_must_be_nfc(self) -> None:
        response_id = new_uuid4()
        composed = "ά"
        decomposed = "α\u0301"
        event = speech_commit(self.factory, response_id, 1, composed)
        self.assertEqual(event.payload["text"], composed)
        with self.assertRaisesRegex(ProtocolError, "protocol.text_not_nfc"):
            speech_commit(self.factory, response_id, 2, decomposed)

    def test_transcript_uses_uncalibrated_asr_score_not_confidence(self) -> None:
        payload = {
            "text": "άνοιξε",
            "language": {"code": "el", "score": 0.97},
            "asr_score": {
                "value": 0.82,
                "method": "exp_weighted_avg_logprob",
                "raw_avg_logprob": -0.2,
                "calibrated": False,
            },
        }
        self.factory.create("input.transcript.final", payload)
        payload["confidence"] = 0.82
        with self.assertRaisesRegex(ProtocolError, "protocol.forbidden_field"):
            self.factory.create("input.transcript.final", payload)


class SequenceTests(unittest.TestCase):
    def test_duplicate_tracking_retains_only_bounded_fingerprints(self) -> None:
        factory = EventFactory()
        tracker = SequenceTracker()
        for text in ("short", "private-transcript-" * 1000):
            event = factory.create("response.text.delta", {"chunk_index": 0, "text": text}, response_id=new_uuid4())
            tracker.observe(event)
            self.assertEqual(len(event.fingerprint()), 64)
            self.assertEqual(tracker.observe(event), SequenceResult.DUPLICATE)
        self.assertNotIn("private-transcript", repr(tracker._seen))

    def setUp(self) -> None:
        self.factory = EventFactory()
        self.tracker = SequenceTracker()

    def test_gap_is_protocol_error(self) -> None:
        first = self.factory.create("server.ready", {"protocol": "jarvis.voice/1"})
        second = self.factory.create(
            "server.error", {"code": "test", "retryable": False, "terminal": True}
        )
        with self.assertRaisesRegex(ProtocolError, "protocol.sequence_gap"):
            self.tracker.observe(second)
        self.assertEqual(self.tracker.observe(first), SequenceResult.ACCEPTED)

    def test_identical_duplicate_is_accepted_as_duplicate(self) -> None:
        event = self.factory.create("server.ready", {"protocol": "jarvis.voice/1"})
        self.assertEqual(self.tracker.observe(event), SequenceResult.ACCEPTED)
        self.assertEqual(
            self.tracker.observe(Envelope.from_dict(event.to_dict())),
            SequenceResult.DUPLICATE,
        )

    def test_same_seq_with_changed_payload_is_conflicting_duplicate(self) -> None:
        response_id = new_uuid4()
        event = self.factory.create(
            "response.text.delta",
            {"chunk_index": 0, "text": "ένα"},
            response_id=response_id,
        )
        self.tracker.observe(event)
        changed = event.to_dict()
        changed["payload"] = {"chunk_index": 0, "text": "δύο"}
        with self.assertRaisesRegex(ProtocolError, "protocol.conflicting_duplicate"):
            self.tracker.observe(Envelope.from_dict(changed))

    def test_reconnect_reuses_event_id_with_new_connection_sequence(self) -> None:
        event_id = new_uuid4()
        original_factory = EventFactory()
        original = original_factory.create("client.goodbye", {}, event_id=event_id)
        original_factory.create("client.goodbye", {})
        reconnect_factory = EventFactory(session_id=original.session_id)
        retransmission = reconnect_factory.create(
            "client.goodbye", {}, event_id=event_id
        )
        self.assertEqual(original.event_id, retransmission.event_id)
        self.assertEqual(retransmission.seq, 1)
        self.assertNotEqual(original.sent_at, "")


if __name__ == "__main__":
    unittest.main()
