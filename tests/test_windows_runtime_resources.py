from __future__ import annotations

import unittest
import queue
import threading
from contextlib import ExitStack
from unittest.mock import Mock, patch

import _bootstrap  # noqa: F401

from jarvis_voice import runtime
from jarvis_voice.errors import BackendError


class WindowsRuntimeResourceTests(unittest.TestCase):
    def test_runtime_surfaces_asynchronous_backend_failures(self):
        for subsystem, code in (
            ("tts", "tts.worker_failed"),
            ("ptt", "input.raw_input_failed"),
        ):
            with self.subTest(subsystem=subsystem):
                instance = runtime.WindowsVoiceRuntime.__new__(
                    runtime.WindowsVoiceRuntime
                )
                instance._stop_requested = threading.Event()
                instance.audio = Mock()
                instance.ptt = Mock(last_error=None)
                instance.transport = Mock()
                instance.core = Mock()
                instance.publisher = Mock(events=queue.Queue())
                instance.publisher.take_forced_release.return_value = None
                instance.approval_keys = Mock(events=queue.Queue())
                instance.callbacks = queue.Queue()
                instance.tts = Mock(last_error=None)
                getattr(instance, subsystem).last_error = BackendError(
                    code, "Backend failed", subsystem=subsystem
                )
                instance.reliable = Mock()
                instance.reliable.due_retransmissions.return_value = []
                instance.reliable.consume_connected_timeout.return_value = False
                instance.cloud_approval = Mock()
                instance.cloud_approval.due_decision_retransmissions.return_value = []
                instance.action_approval = Mock()
                instance.action_approval.due_decision_retransmissions.return_value = []
                instance._receive_or_reconnect = Mock(return_value=None)
                instance.scheduler_runtime = None
                instance._poll_transcription = Mock(
                    side_effect=lambda: setattr(instance, "_running", False)
                )
                instance.close = Mock()
                with self.assertRaisesRegex(BackendError, code):
                    instance.run()
                instance.close.assert_called_once()

    def test_partial_startup_and_failed_shutdown_release_every_owned_resource(self):
        for failure in (
            "pocket",
            "asr",
            "CloudApprovalJournal",
            "SileroOnnxVad",
            "stop",
            "cancel",
            None,
        ):
            for asynchronous in (False, True):
                with (
                    self.subTest(failure=failure, asynchronous=asynchronous),
                    ExitStack() as patches,
                ):
                    config = Mock()
                    config.windows_tts_backend = (
                        "pocket_tts" if failure == "pocket" else "sapi"
                    )
                    config.asr.execution_profile = "cuda" if asynchronous else "cpu"
                    config.scheduler.enabled = True
                    config.ptt_key = "F9"
                    objects = {}
                    patches.enter_context(patch.object(runtime, "PocketTtsCpuEngine"))
                    patches.enter_context(
                        patch.object(
                            runtime,
                            "PocketTtsBackend",
                            side_effect=OSError("pocket startup failed"),
                        )
                    )
                    for name in (
                        "SapiTtsBackend",
                        "DeliveryDedupJournal",
                        "VoiceCore",
                        "CloudApprovalJournal",
                        "CloudApprovalController",
                        "ActionApprovalJournal",
                        "ActionApprovalController",
                        "SileroOnnxVad",
                        "LoopbackWebSocketTransport",
                        "RawInputPttSource",
                        "WasapiAudioCapture",
                        "PttEventPublisher",
                        "ApprovalKeyPublisher",
                    ):
                        obj = objects[name] = Mock()
                        patches.enter_context(
                            patch.object(
                                runtime,
                                name,
                                return_value=obj,
                                side_effect=OSError("startup failed")
                                if name == failure
                                else None,
                            )
                        )
                    for name in (
                        "configured_voice_auth_key",
                        "default_voice_state_boundary",
                        "prepare_voice_journal_state",
                        "default_delivery_journal_path",
                        "default_cloud_approval_journal_path",
                        "default_action_approval_journal_path",
                    ):
                        patches.enter_context(patch.object(runtime, name))
                    scheduler = Mock()
                    patches.enter_context(
                        patch.object(
                            runtime.SchedulerNativeRuntime,
                            "from_config",
                            return_value=scheduler,
                        )
                    )
                    asr = Mock()
                    factory = Mock()
                    factory.create.return_value = asr
                    if failure == "asr":
                        factory.create.side_effect = OSError("asr startup failed")
                    coordinator = Mock()
                    coordinator.close.side_effect = asr.close
                    patches.enter_context(
                        patch.object(
                            runtime,
                            "AsyncTranscriptionCoordinator",
                            return_value=coordinator,
                        )
                    )
                    patches.enter_context(
                        patch.object(runtime.sys, "platform", "win32")
                    )
                    if failure == "stop":
                        objects["RawInputPttSource"].stop.side_effect = OSError(
                            "input stop failed"
                        )
                    if failure == "cancel":
                        objects["SapiTtsBackend"].cancel.side_effect = OSError(
                            "tts cancellation failed"
                        )
                    if failure in (None, "stop", "cancel"):
                        instance = runtime.WindowsVoiceRuntime(
                            config, asr_factory=factory
                        )
                        if failure:
                            with self.assertRaises(OSError):
                                instance.close()
                        else:
                            instance.close()
                        instance.close()
                        objects["WasapiAudioCapture"].stop.assert_called_once()
                        objects["LoopbackWebSocketTransport"].close.assert_called_once()
                    else:
                        with self.assertRaises(OSError):
                            runtime.WindowsVoiceRuntime(config, asr_factory=factory)
                    objects["SapiTtsBackend"].close.assert_called_once()
                    if failure not in ("pocket", "asr"):
                        asr.close.assert_called_once()
                        objects["DeliveryDedupJournal"].close.assert_called_once()
                    if failure not in ("pocket", "asr", "CloudApprovalJournal"):
                        objects["CloudApprovalJournal"].close.assert_called_once()
                        objects["ActionApprovalJournal"].close.assert_called_once()
                        scheduler.close.assert_called_once()
                        if asynchronous:
                            coordinator.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
