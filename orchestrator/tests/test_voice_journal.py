from __future__ import annotations

import tempfile
import sqlite3
import unittest
import uuid
from pathlib import Path
from unittest.mock import Mock, patch

import _bootstrap  # noqa: F401

from jarvis_voice.delivery_journal import DeliveryDedupJournal
from jarvis_voice.errors import ProtocolError
from jarvis_voice.protocol import Envelope, EventFactory
from jarvis_voice.core import VoiceCore
from jarvis_voice.fakes import FakeAsrBackend, FakeTtsBackend


class VoiceJournalTests(unittest.TestCase):
    def test_initial_prune_failure_closes_the_connection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            connection = sqlite3.connect(":memory:")
            self.addCleanup(connection.close)
            with patch("jarvis_voice.delivery_journal.sqlite3.connect", return_value=connection):
                with self.assertRaisesRegex(RuntimeError, "clock unavailable"):
                    DeliveryDedupJournal(
                        Path(directory) / "journal.sqlite3",
                        clock=Mock(side_effect=RuntimeError("clock unavailable")),
                    )
            with self.assertRaises(sqlite3.ProgrammingError):
                connection.execute("SELECT 1")

    def test_restarted_replay_skips_old_unit_and_allows_new_playback(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            journal = DeliveryDedupJournal(Path(directory) / "journal.sqlite3")
            self.addCleanup(journal.close)
            factory = EventFactory()
            response_id = str(uuid.uuid4())
            started = factory.create("response.started", {"locale": "en-US"}, response_id=response_id)
            old = factory.create("response.speech.commit", {
                "unit_index": 1, "locale": "en-US", "text": "Already delivered."
            }, response_id=response_id)
            journal.observe(started)
            journal.observe(old)
            tts = FakeTtsBackend()
            core = VoiceCore(asr=FakeAsrBackend(), tts=tts, delivery_journal=journal)
            self.assertEqual(core.handle_server_events([started, old]), [])
            self.assertTrue(tts.is_idle)
            fresh = factory.create("response.speech.commit", {
                "unit_index": 2, "locale": "en-US", "text": "New speech."
            }, response_id=response_id)
            core.handle_server_event(fresh)
            core.handle_server_event(factory.create("response.speech.done", {
                "last_unit_index": 2
            }, response_id=response_id))
            unit = tts.start_next()
            self.assertEqual(unit.text, "New speech.")
            self.assertEqual([event.type for event in core.on_tts_unit_started(unit)], ["playback.started"])
            self.assertEqual([event.type for event in core.on_tts_unit_completed(tts.complete_active())], ["playback.completed"])

    def test_restarted_replay_can_advance_past_a_finished_response(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            journal = DeliveryDedupJournal(Path(directory) / "journal.sqlite3")
            self.addCleanup(journal.close)
            factory = EventFactory()
            old_id, current_id = str(uuid.uuid4()), str(uuid.uuid4())
            replay = [
                factory.create("response.started", {"locale": "en-US"}, response_id=old_id),
                factory.create("response.speech.done", {"last_unit_index": 0}, response_id=old_id),
                factory.create("response.started", {"locale": "en-US"}, response_id=current_id),
            ]
            for event in replay:
                journal.observe(event)
            tts = FakeTtsBackend()
            core = VoiceCore(asr=FakeAsrBackend(), tts=tts, delivery_journal=journal)
            self.assertEqual(core.handle_server_events(replay), [])
            core.handle_server_event(factory.create("response.speech.commit", {
                "unit_index": 1, "locale": "en-US", "text": "Current response."
            }, response_id=current_id))
            self.assertEqual([unit.text for unit in tts.pending], ["Current response."])

    def test_same_process_replay_preserves_existing_playback_state(self) -> None:
        for phase in ("pending", "active", "completed", "cancelled"):
            with self.subTest(phase=phase), tempfile.TemporaryDirectory() as directory:
                journal = DeliveryDedupJournal(Path(directory) / "journal.sqlite3")
                self.addCleanup(journal.close)
                factory = EventFactory()
                response_id = str(uuid.uuid4())
                events = [
                    factory.create("response.started", {"locale": "en-US"}, response_id=response_id),
                    factory.create("response.speech.commit", {
                        "unit_index": 1, "locale": "en-US", "text": "One sentence."
                    }, response_id=response_id),
                    factory.create("response.speech.done", {"last_unit_index": 1}, response_id=response_id),
                ]
                tts = FakeTtsBackend()
                core = VoiceCore(asr=FakeAsrBackend(), tts=tts, delivery_journal=journal)
                core.handle_server_events(events)
                if phase in ("active", "completed"):
                    core.on_tts_unit_started(tts.start_next())
                if phase == "completed":
                    core.on_tts_unit_completed(tts.complete_active())
                if phase == "cancelled":
                    events.append(factory.create("response.cancelled", {"reason_code": "user_cancelled", "partial_output": False}, response_id=response_id))
                    core.handle_server_event(events[-1])
                original = core.state.response
                core.begin_new_connection()
                self.assertEqual(core.handle_server_events(events), [])
                self.assertIs(core.state.response, original)
                if phase == "pending":
                    core.on_tts_unit_started(tts.start_next())
                if phase in ("pending", "active"):
                    self.assertEqual([event.type for event in core.on_tts_unit_completed(tts.complete_active())], ["playback.completed"])
                else:
                    self.assertTrue(tts.is_idle)

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
