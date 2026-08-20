from __future__ import annotations

import json
import os
import queue
import stat
import struct
import tempfile
import unittest
from pathlib import Path

import _bootstrap  # noqa: F401

from jarvis_voice.private_state import VoicePrivateStateError
from jarvis_voice.macos.runtime import MacVoiceRuntime
from jarvis_voice.ui_state import (
    UI_STATE_VERSION,
    VoiceUiState,
    VoiceUiStatePublisher,
)


@unittest.skipUnless(os.name == "posix", "POSIX owner/mode contract")
class VoiceUiStatePublisherTests(unittest.TestCase):
    def test_publishes_content_free_atomic_snapshot_and_heartbeat(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ticks = iter((10.0, 10.0, 10.5, 10.55, 11.1))
            publisher = VoiceUiStatePublisher(
                root / "state" / "ui-state.json",
                trusted_boundary=root,
                wall_clock=lambda: 1234.5,
                monotonic_clock=lambda: next(ticks),
            )
            publisher.publish(VoiceUiState.LISTENING)
            publisher.observe_input_activity(640)
            publisher.heartbeat()
            before = json.loads(publisher.path.read_text(encoding="ascii"))
            publisher.heartbeat()
            unchanged = json.loads(publisher.path.read_text(encoding="ascii"))
            publisher.heartbeat()
            after = json.loads(publisher.path.read_text(encoding="ascii"))

            self.assertEqual(before["state"], "listening")
            self.assertEqual(before["activity_milli"], 640)
            self.assertEqual(before["version"], UI_STATE_VERSION)
            self.assertEqual(before["updated_at_unix_ms"], 1_234_500)
            self.assertEqual(unchanged, before)
            self.assertGreater(after["generation"], before["generation"])
            self.assertEqual(set(after), {
                "activity_milli", "generation", "state",
                "state_started_at_unix_ms", "updated_at_unix_ms", "version"
            })
            self.assertEqual(stat.S_IMODE(publisher.path.stat().st_mode), 0o600)
            self.assertEqual(stat.S_IMODE(publisher.path.parent.stat().st_mode), 0o700)

    def test_replaces_final_symlink_without_following_it(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state = root / "state"
            state.mkdir(mode=0o700)
            outside = root / "outside"
            outside.write_text("keep", encoding="utf-8")
            outside.chmod(0o600)
            snapshot = state / "ui-state.json"
            snapshot.symlink_to(outside)

            with self.assertRaises(VoicePrivateStateError):
                VoiceUiStatePublisher(snapshot, trusted_boundary=root)

            self.assertTrue(snapshot.is_symlink())
            self.assertEqual(outside.read_text(encoding="utf-8"), "keep")

    def test_rejects_invalid_state_and_never_serializes_arbitrary_content(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            publisher = VoiceUiStatePublisher(
                root / "state" / "ui-state.json",
                trusted_boundary=root,
            )

            with self.assertRaises(TypeError):
                publisher.publish("secret transcript")  # type: ignore[arg-type]
            with self.assertRaises(ValueError):
                publisher.observe_input_activity(1001)

            encoded = publisher.path.read_text(encoding="ascii")
            self.assertNotIn("secret", encoded)
            self.assertNotIn("transcript", encoded)

    def test_runtime_reduces_audio_to_bounded_activity_without_retaining_frames(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            publisher = VoiceUiStatePublisher(
                root / "state" / "ui-state.json",
                trusted_boundary=root,
            )
            publisher.publish(VoiceUiState.LISTENING)
            runtime = MacVoiceRuntime.__new__(MacVoiceRuntime)
            runtime.ui_state = publisher
            runtime._ui_activity_frames = queue.Queue(maxsize=1)
            observed: list[tuple[int, float]] = []
            frame = struct.pack("<64f", *([0.05] * 64))

            runtime._observe_audio_frame(
                frame,
                48_000,
                12.5,
                external_observer=lambda _data, rate, timestamp: observed.append(
                    (rate, timestamp)
                ),
            )
            runtime._drain_ui_activity()
            publisher.publish(VoiceUiState.LISTENING)
            snapshot = json.loads(publisher.path.read_text(encoding="ascii"))

            self.assertEqual(observed, [(48_000, 12.5)])
            self.assertEqual(snapshot["activity_milli"], 400)
            self.assertNotIn("audio", snapshot)
            self.assertNotIn("samples", snapshot)


if __name__ == "__main__":
    unittest.main()
