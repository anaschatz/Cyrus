from __future__ import annotations

import queue
import tempfile
import threading
import unittest
import uuid
from contextlib import closing
from pathlib import Path
from unittest.mock import Mock, patch

import _bootstrap  # noqa: F401
from jarvis_voice import runtime
from jarvis_voice.macos import runtime as mac_runtime
from jarvis_voice.approval_arbiter import LocalApprovalState
from jarvis_voice.cloud_approval import CloudApprovalController, CloudApprovalJournal
from jarvis_voice.state import InputState


class ApprovalRuntimeRearmTests(unittest.TestCase):
    def test_both_runtime_loops_rearm_only_when_idle_and_keys_released(self):
        for runtime_class in (runtime.WindowsVoiceRuntime, mac_runtime.MacVoiceRuntime):
            for input_state, keys_up in ((InputState.IDLE, True), (InputState.CAPTURING, True), (InputState.IDLE, False)):
                with self.subTest(runtime=runtime_class.__name__, input_state=input_state, keys_up=keys_up), tempfile.TemporaryDirectory() as directory, closing(CloudApprovalJournal(Path(directory) / "cloud.sqlite3")) as journal:
                    cloud = CloudApprovalController(session_id=str(uuid.uuid4()), journal=journal)
                    arbiter = cloud.approval_arbiter
                    arbiter.stage(protocol="jarvis.action-approval/1", approval_id=str(uuid.uuid4()),
                                  presentation_id=str(uuid.uuid4()), requested_event_id=str(uuid.uuid4()),
                                  key_counter=1, both_keys_up=True)
                    arbiter.disarm_for_capture()
                    instance = runtime_class.__new__(runtime_class)
                    instance._stop_requested = threading.Event()
                    instance.audio = Mock(is_healthy=True)
                    instance.ptt = Mock(last_error=None)
                    instance.asr = Mock()
                    instance.vad = Mock()
                    instance.transport = Mock()
                    instance.core = Mock()
                    instance.core.state.input_state = input_state
                    instance.publisher = Mock(events=queue.Queue())
                    instance.publisher.take_forced_release.return_value = None
                    instance.approval_keys = Mock(events=queue.Queue(), counter=5, both_keys_up=keys_up)
                    instance.callbacks = queue.Queue()
                    instance.tts = Mock(last_error=None)
                    instance.reliable = Mock()
                    instance.reliable.due_retransmissions.return_value = []
                    instance.reliable.consume_connected_timeout.return_value = False
                    instance.cloud_approval = cloud
                    instance.action_approval = Mock()
                    instance.action_approval.due_decision_retransmissions.return_value = []
                    instance._receive_or_reconnect = Mock(return_value=None)
                    instance.scheduler_runtime = None
                    instance._poll_transcription = Mock(side_effect=lambda: setattr(instance, "_running", False))
                    instance.recovery = Mock()
                    instance.recovery.poll.return_value = None
                    for method in ("close", "_drain_ui_activity", "_heartbeat_ui_state", "_record_timing", "_publish_ui_state"):
                        setattr(instance, method, Mock())
                    with patch.object(mac_runtime, "warm_macos_asr_backend", return_value=None), patch.object(mac_runtime, "warm_macos_vad_backend", return_value=0):
                        instance.run()
                    rearmed = input_state == InputState.IDLE and keys_up
                    self.assertEqual(arbiter.presentation.state,
                                     LocalApprovalState.ARMED if rearmed else LocalApprovalState.DISARMED_CAPTURE)
                    self.assertEqual(arbiter.presentation.arm_counter, 5 if rearmed else 1)
                    instance.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
