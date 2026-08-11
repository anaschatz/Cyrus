from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
import uuid
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import _bootstrap  # noqa: F401

from jarvis_voice.backends import SpeechUnit
from jarvis_voice.approval_arbiter import GlobalApprovalController
from jarvis_voice.errors import ProtocolError
from jarvis_voice.config import voice_config_from_mapping
from jarvis_voice.fakes import FakeTtsBackend
from jarvis_voice.native_qualification import native_evidence_template
from jarvis_voice.scheduler_journal import SchedulerClientJournal
from jarvis_voice.scheduler_client import SchedulerClientController
from jarvis_voice.scheduler_playback import SchedulerPlaybackController, SchedulerPlaybackRecord
from jarvis_voice.scheduler_compatibility import SchedulerCompatibilityManifest
from jarvis_voice.scheduler_protocol import SchedulerEventFactory
from jarvis_voice.scheduler_runtime import SchedulerNativeRuntime
from jarvis_voice.scheduler_tts_qualification import load_scheduler_tts_qualification
from jarvis_voice.state import VoiceStateMachine
from jarvis_orchestrator.scheduler_crypto import InMemorySchedulerKeyProvider, SchedulerContentCipher


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

    def cancel_notification_playback(self, notification_id: str) -> list[int]:
        cancelled = [
            unit_index
            for (selected, unit_index), state in self.states.items()
            if selected == notification_id and state in {"STAGED", "PLAYING"}
        ]
        for unit_index in cancelled:
            self.states[(notification_id, unit_index)] = "CANCELLED"
        return sorted(cancelled)


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

    def test_ptt_cancels_remaining_units_of_same_notification(self) -> None:
        state = VoiceStateMachine()
        journal = FakePlaybackJournal([record(1), record(2)])
        tts = FakeTtsBackend()
        controller = self.controller(journal, tts, state)
        controller.pump(allowed=True)
        receipts = controller.cancel_active(
            cancellation_epoch=state.output_epoch + 1,
            reason_code="ptt_down",
        )
        self.assertEqual([item.payload["unit_index"] for item in receipts], [1, 2])
        self.assertEqual(set(journal.states.values()), {"CANCELLED"})

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

    def test_scheduler_tts_capability_requires_digest_bound_ready_native_evidence(self) -> None:
        evidence = native_evidence_template("macos")
        evidence["provenance"] = {
            "native_machine": True,
            "voice_layer_mocks_used": False,
            "synthetic_voice_turns": False,
            "hardware_capture": True,
            "physical_ptt": True,
            "physical_playback": True,
        }
        evidence["cases"] = {
            name: {"status": "pass", "attempts": 1} for name in evidence["cases"]
        }
        evidence["counters"] = {
            "real_voice_turns": 200,
            "duplicate_transcripts": 0,
            "duplicate_tts_playbacks": 0,
            "stuck_ptt": 0,
            "microphone_disconnect_recoveries": 1,
            "sleep_resume_recoveries": 1,
            "crashes": 0,
        }
        evidence["acoustic_barge_in"] = {
            "method": "acoustic_microphone",
            "audio_path": "built_in",
            "samples": 20,
            "max_ptt_to_silence_ms": 120.0,
            "silence_hold_ms": 50,
        }
        evidence["soak"] = {
            "duration_seconds": 28_800,
            "rss_samples": 97,
            "rss_start_bytes": 100,
            "rss_end_bytes": 100,
            "rss_peak_bytes": 100,
            "rss_slope_bytes_per_hour": 0.0,
        }
        evidence["runtime_profile"] = {
            "mode": "source",
            "fingerprint_sha256": "f" * 64,
            "stable": True,
            "signed": False,
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "native.json"
            raw = json.dumps(evidence, sort_keys=True, separators=(",", ":")).encode()
            path.write_bytes(raw)
            digest = hashlib.sha256(raw).hexdigest()
            compatibility = SchedulerCompatibilityManifest(
                "e" * 64,
                ("jarvis.scheduler/1", "jarvis.approval-arbiter/1", "scheduler_notification_speech"),
                {"tts_adapter_qualification_sha256": digest},
            )
            loaded = load_scheduler_tts_qualification(
                path,
                digest,
                platform_name="darwin",
                compatibility=compatibility,
            )
            self.assertEqual(loaded.runtime_profile_sha256, "f" * 64)
            with self.assertRaisesRegex(ValueError, "digest mismatch"):
                load_scheduler_tts_qualification(
                    path,
                    "0" * 64,
                    platform_name="darwin",
                    compatibility=compatibility,
                )

    def test_scheduler_tts_config_must_be_an_absolute_digest_pair(self) -> None:
        base = {
            "enabled": True,
            "compatibility_manifest_path": "/private/compatibility.json",
            "compatibility_manifest_sha256": "a" * 64,
            "native_qualification_path": "/private/native.json",
            "native_qualification_sha256": "b" * 64,
        }
        with self.assertRaisesRegex(Exception, "scheduler_tts_qualification_incomplete"):
            voice_config_from_mapping(
                {"scheduler": {**base, "tts_qualification_path": "/private/voice-native.json"}}
            )
        with self.assertRaisesRegex(Exception, "scheduler_path_not_absolute"):
            voice_config_from_mapping(
                {
                    "scheduler": {
                        **base,
                        "tts_qualification_path": "relative.json",
                        "tts_qualification_sha256": "c" * 64,
                    }
                }
            )


class SchedulerPlaybackJournalTests(unittest.TestCase):
    @staticmethod
    def _payload(notification_id: str, unit_index: int, text: str) -> dict[str, object]:
        return {
            "occurrence_id": "b" * 64,
            "notification_id": notification_id,
            "unit_index": unit_index,
            "locale": "el-GR",
            "speakable_text": text,
        }

    def test_gap_conflict_durable_order_and_whole_notification_cancel(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            journal = SchedulerClientJournal(
                Path(temporary) / "scheduler.sqlite3",
                cipher=SchedulerContentCipher(InMemorySchedulerKeyProvider()),
            )
            notification_id = "a" * 64
            second_id = str(uuid.uuid4())
            journal.stage(
                event_id=second_id,
                semantic_sha256="2" * 64,
                event_type="scheduler.notification.speech",
                payload=self._payload(notification_id, 2, "Δεύτερο."),
            )
            with self.assertRaisesRegex(ProtocolError, "scheduler_speech_gap"):
                journal.stage_playback(
                    notification_id=notification_id,
                    unit_index=2,
                    frame_event_id=second_id,
                )
            first_id = str(uuid.uuid4())
            journal.stage(
                event_id=first_id,
                semantic_sha256="1" * 64,
                event_type="scheduler.notification.speech",
                payload=self._payload(notification_id, 1, "Πρώτο."),
            )
            self.assertTrue(journal.stage_playback(
                notification_id=notification_id,
                unit_index=1,
                frame_event_id=first_id,
            ))
            conflict_id = str(uuid.uuid4())
            journal.stage(
                event_id=conflict_id,
                semantic_sha256="3" * 64,
                event_type="scheduler.notification.speech",
                payload=self._payload(notification_id, 1, "Άλλο."),
            )
            with self.assertRaisesRegex(ProtocolError, "conflicting_scheduler_speech"):
                journal.stage_playback(
                    notification_id=notification_id,
                    unit_index=1,
                    frame_event_id=conflict_id,
                )
            self.assertTrue(journal.stage_playback(
                notification_id=notification_id,
                unit_index=2,
                frame_event_id=second_id,
            ))
            first = journal.next_staged_playback()
            self.assertEqual((first.unit_index, first.speakable_text), (1, "Πρώτο."))
            self.assertTrue(journal.set_playback_state(notification_id, 1, "PLAYING"))
            self.assertEqual(journal.cancel_notification_playback(notification_id), [1, 2])
            self.assertFalse(journal.set_playback_state(notification_id, 1, "COMPLETED"))
            journal.close()

    def test_scheduler_runtime_stages_before_shared_tts_and_routes_receipts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            journal = SchedulerClientJournal(
                Path(temporary) / "scheduler.sqlite3",
                cipher=SchedulerContentCipher(InMemorySchedulerKeyProvider()),
            )
            session_id = str(uuid.uuid4())
            controller = SchedulerClientController(
                session_id=session_id,
                journal=journal,
                approval_arbiter=GlobalApprovalController(),
            )
            state = VoiceStateMachine()
            tts = FakeTtsBackend()
            playback = SchedulerPlaybackController(
                journal=journal,
                tts=tts,
                factory=controller.factory,
                reserve_epoch=state.reserve_output_epoch,
                utc_now=lambda: datetime(2026, 8, 11, tzinfo=UTC),
            )
            runtime = SchedulerNativeRuntime(
                controller=controller,
                delivery=SimpleNamespace(),
                capabilities=(
                    "jarvis.scheduler/1",
                    "jarvis.approval-arbiter/1",
                    "scheduler_notification_speech",
                ),
                manifest_sha256="d" * 64,
                playback=playback,
            )
            event = SchedulerEventFactory(session_id).create(
                "scheduler.notification.speech",
                self._payload("a" * 64, 1, "Υπενθύμιση."),
            )
            sent = []
            runtime.handle_server_event(
                event,
                key_counter=0,
                both_keys_up=True,
                send=sent.append,
            )
            self.assertEqual([item.type for item in sent], ["scheduler.event_ack"])
            self.assertFalse(tts.pending)
            self.assertEqual(runtime.pump_playback(allowed=True), [])
            unit = tts.pending[0]
            handled, started = runtime.handle_tts_callback("started", unit)
            self.assertTrue(handled)
            self.assertEqual(started[0].payload["state"], "PLAYING")
            tts.start_next()
            tts.complete_active()
            handled, completed = runtime.handle_tts_callback("completed", unit)
            self.assertTrue(handled)
            self.assertEqual(completed[0].payload["state"], "COMPLETED")
            runtime.close()

    def test_restart_converts_playing_to_cancelled_once(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            journal = SchedulerClientJournal(
                Path(temporary) / "scheduler.sqlite3",
                cipher=SchedulerContentCipher(InMemorySchedulerKeyProvider()),
            )
            notification_id = "c" * 64
            event_id = str(uuid.uuid4())
            journal.stage(
                event_id=event_id,
                semantic_sha256="4" * 64,
                event_type="scheduler.notification.speech",
                payload=self._payload(notification_id, 1, "Υπενθύμιση."),
            )
            journal.stage_playback(
                notification_id=notification_id,
                unit_index=1,
                frame_event_id=event_id,
            )
            journal.set_playback_state(notification_id, 1, "PLAYING")
            self.assertEqual(journal.recover_interrupted_playback(), [(notification_id, 1)])
            self.assertEqual(journal.recover_interrupted_playback(), [])
            journal.close()


if __name__ == "__main__":
    unittest.main()
