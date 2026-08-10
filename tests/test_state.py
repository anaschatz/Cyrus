from __future__ import annotations

import unittest

import _bootstrap  # noqa: F401

from jarvis_voice.backends import SpeechUnit
from jarvis_voice.errors import ProtocolError, StateTransitionError
from jarvis_voice.protocol import new_uuid4
from jarvis_voice.state import ApplyResult, InputState, ResponseSession, VoiceStateMachine


def unit(response: ResponseSession, index: int, text: str | None = None) -> SpeechUnit:
    return SpeechUnit(
        response_id=response.response_id,
        epoch=response.epoch,
        unit_index=index,
        locale="el-GR",
        text=text or f"Πρόταση {index}.",
    )


class ResponseSessionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.response = ResponseSession(new_uuid4(), epoch=1)

    def test_units_start_at_one_and_progress_without_gaps(self) -> None:
        with self.assertRaisesRegex(ProtocolError, "protocol.speech_unit_gap"):
            self.response.add_commit(unit(self.response, 2))
        self.assertEqual(self.response.add_commit(unit(self.response, 1)), ApplyResult.ACCEPTED)
        self.assertEqual(self.response.add_commit(unit(self.response, 2)), ApplyResult.ACCEPTED)

    def test_identical_and_conflicting_duplicate_commit(self) -> None:
        first = unit(self.response, 1)
        self.response.add_commit(first)
        self.assertEqual(self.response.add_commit(first), ApplyResult.DUPLICATE)
        with self.assertRaisesRegex(ProtocolError, "protocol.conflicting_speech_commit"):
            self.response.add_commit(unit(self.response, 1, "Διαφορετικό."))

    def test_speech_done_requires_all_units_and_is_idempotent(self) -> None:
        self.response.add_commit(unit(self.response, 1))
        with self.assertRaisesRegex(ProtocolError, "protocol.speech_done_gap"):
            self.response.mark_speech_done(2)
        self.assertEqual(self.response.mark_speech_done(1), ApplyResult.ACCEPTED)
        self.assertEqual(self.response.mark_speech_done(1), ApplyResult.DUPLICATE)
        with self.assertRaisesRegex(ProtocolError, "protocol.conflicting_speech_done"):
            self.response.mark_speech_done(0)

    def test_commit_after_speech_done_is_rejected(self) -> None:
        self.response.mark_speech_done(0)
        with self.assertRaisesRegex(ProtocolError, "protocol.commit_after_speech_done"):
            self.response.add_commit(unit(self.response, 1))

    def test_playback_completion_requires_every_condition(self) -> None:
        first = unit(self.response, 1)
        self.response.add_commit(first)
        self.assertFalse(self.response.can_complete_playback(backend_idle=True))
        self.response.mark_speech_done(1)
        self.assertFalse(self.response.can_complete_playback(backend_idle=True))
        self.response.mark_unit_started(1)
        self.assertFalse(self.response.can_complete_playback(backend_idle=False))
        self.response.mark_unit_completed(1)
        self.assertFalse(self.response.can_complete_playback(backend_idle=False))
        self.assertTrue(self.response.can_complete_playback(backend_idle=True))

    def test_empty_response_can_complete_after_done_zero(self) -> None:
        self.assertEqual(self.response.mark_speech_done(0), ApplyResult.ACCEPTED)
        self.assertTrue(self.response.can_complete_playback(backend_idle=True))

    def test_tombstone_wins_and_interruption_is_emitted_once(self) -> None:
        self.response.add_commit(unit(self.response, 1))
        self.response.mark_speech_done(1)
        self.assertTrue(self.response.tombstone(cancelled=True))
        self.assertFalse(self.response.tombstone(cancelled=True))
        self.assertEqual(self.response.mark_speech_done(1), ApplyResult.STALE)
        self.assertEqual(self.response.mark_unit_started(1), ApplyResult.STALE)
        self.assertFalse(self.response.can_complete_playback(backend_idle=True))

    def test_completion_without_start_is_invalid(self) -> None:
        self.response.add_commit(unit(self.response, 1))
        with self.assertRaises(StateTransitionError):
            self.response.mark_unit_completed(1)


class VoiceStateMachineTests(unittest.TestCase):
    def test_input_lifecycle(self) -> None:
        state = VoiceStateMachine()
        turn_id = new_uuid4()
        state.ptt_down(turn_id)
        self.assertEqual(state.input_state, InputState.CAPTURING)
        state.ptt_up()
        state.transcription_complete()
        state.submission_complete()
        self.assertEqual(state.input_state, InputState.IDLE)

    def test_ptt_cancels_before_and_after_speech_done(self) -> None:
        for done in (False, True):
            state = VoiceStateMachine()
            response = state.start_response(new_uuid4())
            state.add_speech_commit(response.response_id, 1, "el-GR", "Ναι.")
            if done:
                state.mark_speech_done(response.response_id, 1)
            result = state.ptt_down(new_uuid4())
            self.assertEqual(result.cancel_response_id, response.response_id)
            self.assertTrue(result.emit_playback_interrupted)
            self.assertTrue(response.tombstoned)
            self.assertFalse(state.maybe_complete_playback(backend_idle=True))

    def test_late_callback_from_old_epoch_is_ignored(self) -> None:
        state = VoiceStateMachine()
        response = state.start_response(new_uuid4())
        _, old_unit = state.add_speech_commit(
            response.response_id, 1, "el-GR", "Παλιά απάντηση."
        )
        assert old_unit is not None
        state.ptt_down(new_uuid4())
        self.assertEqual(state.mark_unit_started(old_unit), ApplyResult.STALE)
        self.assertEqual(state.mark_unit_completed(old_unit), ApplyResult.STALE)

    def test_cancel_done_race_favors_tombstone(self) -> None:
        state = VoiceStateMachine()
        response = state.start_response(new_uuid4())
        state.add_speech_commit(response.response_id, 1, "en-US", "Done.")
        self.assertTrue(state.cancel_response(response.response_id))
        self.assertEqual(
            state.mark_speech_done(response.response_id, 1), ApplyResult.STALE
        )
        self.assertFalse(state.maybe_complete_playback(backend_idle=True))


if __name__ == "__main__":
    unittest.main()
