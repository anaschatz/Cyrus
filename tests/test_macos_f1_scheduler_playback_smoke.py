from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path
from unittest.mock import patch

import _bootstrap  # noqa: F401

from jarvis_voice.config import voice_config_from_mapping
from jarvis_voice.errors import BackendError


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_macos_f1_scheduler_playback_smoke.py"
SPEC = importlib.util.spec_from_file_location("run_macos_f1_scheduler_playback_smoke", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class FakeTts:
    def __init__(self, on_started, on_completed):
        self.on_started = on_started
        self.on_completed = on_completed
        self.is_idle = True
        self.last_error = None
        self.closed = False

    def enqueue(self, unit):
        self.on_started(unit)
        self.on_completed(unit)

    def cancel(self, epoch, reason):
        del epoch, reason

    def close(self):
        self.closed = True


class MacosF1SchedulerPlaybackSmokeTests(unittest.TestCase):
    def test_invalid_timeout_never_starts_tts(self) -> None:
        for timeout in (float("nan"), float("inf"), 0.0, -1.0):
            with self.subTest(timeout=timeout), patch.object(
                MODULE, "create_macos_calibration_tts", side_effect=AssertionError("native access")
            ) as build:
                with self.assertRaises(BackendError) as caught:
                    MODULE.run_scheduler_playback_smoke(
                        voice_config_from_mapping({}), timeout_seconds=timeout, tts_builder=build,
                    )
                self.assertEqual(caught.exception.code, "qualification.scheduler_playback_config_invalid")
                build.assert_not_called()

    def test_enqueue_failure_is_reported_without_waiting_for_a_callback(self) -> None:
        holder = {}

        def build(_config, *, on_started, on_completed):
            tts = FakeTts(on_started, on_completed)
            holder["tts"] = tts
            def fail_enqueue(_unit):
                raise RuntimeError("enqueue failed")

            tts.enqueue = fail_enqueue
            return tts

        with patch.object(MODULE.queue.Queue, "get", side_effect=AssertionError("must not wait")):
            with self.assertRaises(BackendError) as caught:
                MODULE.run_scheduler_playback_smoke(voice_config_from_mapping({}), tts_builder=build)
        self.assertEqual(caught.exception.code, "qualification.scheduler_playback_tts_failed")
        self.assertTrue(holder["tts"].closed)

    def test_real_controller_emits_started_and_completed_receipts(self) -> None:
        holder = {}
        now = [1.0]

        def clock() -> float:
            value = now[0]
            now[0] += 0.01
            return value

        def build(_config, *, on_started, on_completed):
            holder["tts"] = FakeTts(on_started, on_completed)
            return holder["tts"]

        report = MODULE.run_scheduler_playback_smoke(
            voice_config_from_mapping({}),
            tts_builder=build,
            clock=clock,
        )
        self.assertTrue(report["ready"])
        self.assertEqual(report["receipts"], ["PLAYING", "COMPLETED"])
        self.assertEqual(report["journal_state"], "COMPLETED")
        self.assertTrue(holder["tts"].closed)


if __name__ == "__main__":
    unittest.main()
