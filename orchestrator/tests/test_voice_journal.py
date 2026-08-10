from __future__ import annotations

import tempfile
import unittest
import uuid
from pathlib import Path

import _bootstrap  # noqa: F401

from jarvis_voice.delivery_journal import DeliveryDedupJournal
from jarvis_voice.errors import ProtocolError
from jarvis_voice.protocol import Envelope, EventFactory
from jarvis_voice.core import VoiceCore
from jarvis_voice.fakes import FakeAsrBackend, FakeTtsBackend


class VoiceJournalTests(unittest.TestCase):
    def event(self, *, event_id: str | None = None, text: str = "Γεια") -> Envelope:
        return EventFactory().create(
            "response.speech.commit",
            {"unit_index": 1, "locale": "el-GR", "text": text},
            response_id=str(uuid.uuid4()),
            event_id=event_id,
        )

    def test_restart_suppresses_same_logical_event_and_detects_conflict(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "journal.sqlite3"
            event = self.event()
            first = DeliveryDedupJournal(path)
            self.assertFalse(first.observe(event))
            first.close()
            second = DeliveryDedupJournal(path)
            self.addCleanup(second.close)
            self.assertTrue(second.observe(Envelope.from_dict(event.to_dict())))
            changed = event.to_dict()
            changed["payload"] = {"unit_index": 1, "locale": "el-GR", "text": "Άλλο"}
            with self.assertRaisesRegex(ProtocolError, "protocol.conflicting_duplicate"):
                second.observe(Envelope.from_dict(changed))

    def test_live_capacity_fails_closed_and_expiry_prunes(self) -> None:
        now = [100.0]
        with tempfile.TemporaryDirectory() as directory:
            journal = DeliveryDedupJournal(Path(directory) / "journal.sqlite3", maximum_rows=1, ttl_seconds=60, clock=lambda: now[0])
            self.addCleanup(journal.close)
            journal.observe(self.event())
            with self.assertRaisesRegex(ProtocolError, "delivery_journal.capacity_exceeded"):
                journal.observe(self.event())
            now[0] = 161.0
            self.assertEqual(journal.prune(), 1)
            self.assertFalse(journal.observe(self.event()))

    def test_crash_after_started_fence_rebuilds_state_without_duplicate_tts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "journal.sqlite3"
            response_id = str(uuid.uuid4())
            factory = EventFactory()
            started = factory.create("response.started", {"locale": "el-GR"}, response_id=response_id)
            first = DeliveryDedupJournal(path)
            self.assertFalse(first.observe(started))  # persisted, process dies before core application
            first.close()
            journal = DeliveryDedupJournal(path)
            self.addCleanup(journal.close)
            tts = FakeTtsBackend()
            core = VoiceCore(asr=FakeAsrBackend(), tts=tts, delivery_journal=journal)
            core.handle_server_event(Envelope.from_dict(started.to_dict()))
            commit = factory.create(
                "response.speech.commit",
                {"unit_index": 1, "locale": "el-GR", "text": "Νέα μονάδα."},
                response_id=response_id,
            )
            core.handle_server_event(commit)
            self.assertEqual([unit.text for unit in tts.pending], ["Νέα μονάδα."])


if __name__ == "__main__":
    unittest.main()
