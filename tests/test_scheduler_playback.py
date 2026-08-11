from __future__ import annotations

import unittest
import uuid
from datetime import UTC, datetime

import _bootstrap  # noqa: F401

from jarvis_voice.backends import SpeechUnit
from jarvis_voice.errors import ProtocolError
from jarvis_voice.fakes import FakeTtsBackend
from jarvis_voice.scheduler_playback import SchedulerPlaybackController, SchedulerPlaybackRecord
from jarvis_voice.scheduler_protocol import SchedulerEventFactory
from jarvis_voice.state import VoiceStateMachine


class FakePlaybackJournal:
    def __init__(self, records: list[SchedulerPlaybackRecord]) -> None:
        self.records = list(records)
        self.states = {(item.notification_id, item.unit_index): "STAGED" for item in records}
        self.interrupted: list[tuple[str, int]] = []

    def next_staged_playback(self) -> SchedulerPlaybackRecord | None:
        return next(
            (item for item in self.records if self.states[(item.notification_id, item.unit_index)] == "STAGED"),
            None,
        )

    def set_playback_state(self, notification_id: str, unit_index: int, state: str) -> bool:
        key = (notification_id, unit_index)
        current = self.states.get(key)
        allowed = {
            "PLAYING": {"STAGED"},
            "COMPLETED": {"PLAYING"},
            "CANCELLED": {"STAGED", "PLAYING"},
            "SKIPPED": {"STAGED", "PLAYING"},
        }
        if current not in allowed[state]:
            return False
        self.states[key] = state
        return True

    def recover_interrupted_playback(self) -> list[tuple[str, int]]:
        result = list(self.interrupted)
        self.interrupted.clear()
        return result


def record(index: int = 1) -> SchedulerPlaybackRecord:
    return SchedulerPlaybackRecord("a" * 64, index, "el-GR", "Υπενθύμιση.")


class SchedulerPlaybackTests(unittest.TestCase):
    def controller(self, journal: FakePlaybackJournal, tts: FakeTtsBackend, state: VoiceStateMachine) -> SchedulerPlaybackController:
        return SchedulerPlaybackController(
            journal=journal,
            tts=tts,
            factory=SchedulerEventFactory(str(uuid.uuid4())),
            reserve_epoch=state.reserve_output_epoch,
            utc_now=lambda: datetime(2026, 8, 11, tzinfo=UTC),
        )

    def test_scheduler_unit_uses_shared_fence_but_not_conversation_response(self) -> None:
        state = VoiceStateMachine()
        journal = FakePlaybackJournal([record()])
        tts = FakeTtsBackend()
        controller = self.controller(journal, tts, state)
        self.assertEqual(controller.pump(allowed=True), [])
        unit = tts.pending[0]
        self.assertTrue(controller.owns(unit))
        self.assertEqual((unit.epoch, state.output_epoch), (1, 1))
        self.assertIsNone(state.response)
        handled, started = controller.on_started(unit)
        self.assertTrue(handled)
        self.assertEqual(started[0].payload["state"], "PLAYING")
        tts.start_next()
        tts.complete_active()
        handled, completed = controller.on_completed(unit)
        self.assertTrue(handled)
        self.assertEqual(completed[0].payload["state"], "COMPLETED")
        self.assertEqual(journal.states[("a" * 64, 1)], "COMPLETED")

    def test_ptt_cancellation_wins_and_late_callbacks_are_stale(self) -> None:
        state = VoiceStateMachine()
        journal = FakePlaybackJournal([record()])
        tts = FakeTtsBackend()
        controller = self.controller(journal, tts, state)
        controller.pump(allowed=True)
        unit = tts.pending[0]
        cancel_epoch = state.output_epoch + 1
        receipts = controller.cancel_active(cancellation_epoch=cancel_epoch, reason_code="ptt_down")
        self.assertEqual(receipts[0].payload["state"], "CANCELLED")
        self.assertEqual(tts.cancelled_through_epoch, cancel_epoch)
        self.assertTrue(controller.on_started(unit)[0])
        self.assertEqual(controller.on_completed(unit), (True, []))
        self.assertEqual(journal.states[("a" * 64, 1)], "CANCELLED")

    def test_conversation_activity_blocks_scheduler_and_restart_is_receipted(self) -> None:
        state = VoiceStateMachine()
        journal = FakePlaybackJournal([record()])
        journal.interrupted = [("b" * 64, 2)]
        tts = FakeTtsBackend()
        controller = self.controller(journal, tts, state)
        self.assertEqual(controller.pump(allowed=False), [])
        self.assertFalse(tts.pending)
        recovered = controller.startup_recovery()
        self.assertEqual(recovered[0].payload["reason_code"], "restart_interrupted_playback")

    def test_non_scheduler_callbacks_are_left_for_conversation_core(self) -> None:
        controller = self.controller(FakePlaybackJournal([]), FakeTtsBackend(), VoiceStateMachine())
        unit = SpeechUnit(str(uuid.uuid4()), 1, 1, "en-US", "Hello.")
        self.assertEqual(controller.on_started(unit), (False, []))
        self.assertEqual(controller.on_completed(unit), (False, []))

    def test_scheduler_speech_and_receipt_payloads_are_strict(self) -> None:
        factory = SchedulerEventFactory(str(uuid.uuid4()))
        with self.assertRaisesRegex(ProtocolError, "invalid scheduler speech unit"):
            factory.create(
                "scheduler.notification.speech",
                {
                    "occurrence_id": "b" * 64,
                    "notification_id": "a" * 64,
                    "unit_index": 0,
                    "locale": "el-GR",
                    "speakable_text": "Υπενθύμιση.",
                },
            )
        with self.assertRaisesRegex(ProtocolError, "invalid scheduler playback receipt"):
            factory.create(
                "scheduler.playback.receipt",
                {
                    "notification_id": "a" * 64,
                    "unit_index": 1,
                    "state": "DONE",
                    "reason_code": "free form",
                    "observed_at_utc": "2026-08-11T00:00:00.000Z",
                },
            )


if __name__ == "__main__":
    unittest.main()
