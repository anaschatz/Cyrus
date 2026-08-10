from __future__ import annotations

import json
import unittest

import _bootstrap  # noqa: F401

from jarvis_voice.errors import ProtocolError
from jarvis_voice.protocol import Envelope, EventFactory, new_uuid4
from jarvis_voice.sequencing import SequenceResult, SequenceTracker

from helpers import speech_commit


class EnvelopeTests(unittest.TestCase):
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
