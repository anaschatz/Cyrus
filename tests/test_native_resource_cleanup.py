from __future__ import annotations

import ctypes
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import _bootstrap  # noqa: F401

from jarvis_voice.errors import BackendError
from jarvis_voice.macos.audio_capture import CoreAudioCapture
from jarvis_voice.windows.audio_capture import WasapiAudioCapture
from jarvis_voice.windows.hardening import Win32DllDirectoryApi
from jarvis_voice.windows.raw_input import RawInputPttSource
from jarvis_voice.windows.single_instance import Win32MutexApi
from jarvis_voice.windows.tray import WindowsTrayIcon


class NativeResourceCleanupTests(unittest.TestCase):
    def test_raw_input_worker_exit_is_observable(self):
        for error, stopping, code in (
            (None, True, None),
            (None, False, "input.raw_input_closed"),
            (OSError("private failure"), False, "input.raw_input_failed"),
            (
                BackendError("input.test_failure", "test", subsystem="input"),
                False,
                "input.test_failure",
            ),
        ):
            with self.subTest(code=code):
                source = RawInputPttSource(Mock())
                source._run_win32_loop = Mock(side_effect=error)
                if stopping:
                    source._stop_requested.set()
                source._message_loop()
                self.assertTrue(source._ready.is_set())
                self.assertEqual(
                    source.last_error.code if source.last_error else None, code
                )

    def test_raw_input_restart_resets_handshake_and_retains_live_worker(self):
        source = RawInputPttSource(Mock())
        source._error = OSError("previous startup failed")
        source._ready.set()
        old_thread = Mock()
        old_thread.is_alive.return_value = False
        source._thread = old_thread
        new_thread = Mock()

        def start():
            self.assertFalse(source._ready.is_set())
            self.assertIsNone(source._error)
            source._hwnd = 123
            source._ready.set()

        new_thread.start.side_effect = start
        with (
            patch.object(sys, "platform", "win32"),
            patch(
                "jarvis_voice.windows.raw_input.threading.Thread",
                return_value=new_thread,
            ),
            patch.object(ctypes, "windll", SimpleNamespace(user32=Mock()), create=True),
        ):
            source.start()
            source.stop()
            self.assertIs(source._thread, new_thread)
            with self.assertRaisesRegex(BackendError, "input.raw_input_stopping"):
                source.start()
            new_thread.is_alive.return_value = False
            source.stop()
            self.assertIsNone(source._thread)

    def test_audio_closes_after_failed_start_and_stop(self):
        for capture_type, platform in (
            (CoreAudioCapture, "darwin"),
            (WasapiAudioCapture, "win32"),
        ):
            for failure in ("start", "stop", None):
                with self.subTest(capture=capture_type.__name__, failure=failure):
                    stream = Mock()
                    if failure:
                        getattr(stream, failure).side_effect = OSError(
                            "private device error"
                        )
                    sounddevice = SimpleNamespace(
                        query_devices=Mock(return_value={"default_samplerate": 48000}),
                        RawInputStream=Mock(return_value=stream),
                        CoreAudioSettings=Mock(),
                        WasapiSettings=Mock(),
                    )
                    capture = capture_type(
                        queue_capacity=4, maximum_utterance_seconds=1
                    )
                    with (
                        patch.dict(sys.modules, {"sounddevice": sounddevice}),
                        patch.object(sys, "platform", platform),
                    ):
                        if failure == "start":
                            with self.assertRaises(BackendError):
                                capture.start()
                        else:
                            capture.start()
                            capture.start()
                            capture.stop()
                        capture.stop()
                    sounddevice.RawInputStream.assert_called_once()
                    stream.close.assert_called_once()
                    with self.assertRaisesRegex(BackendError, "audio.not_started"):
                        capture.begin_turn()

    def test_native_wrappers_report_saved_error(self):
        kernel32 = SimpleNamespace(
            SetDefaultDllDirectories=Mock(return_value=0),
            CloseHandle=Mock(return_value=0),
        )
        for operation in (
            lambda: Win32DllDirectoryApi().set_default_directories(0x1000),
            lambda: Win32MutexApi().close(9),
        ):
            with (
                self.subTest(operation=operation),
                patch.object(ctypes, "WinDLL", return_value=kernel32, create=True),
                patch.object(ctypes, "get_last_error", return_value=5, create=True),
                patch.object(
                    ctypes,
                    "WinError",
                    return_value=OSError(5, "native failure"),
                    create=True,
                ) as error,
                self.assertRaises(OSError),
            ):
                try:
                    operation()
                finally:
                    error.assert_called_once_with(5)

    def test_native_windows_and_classes_are_released_before_restart(self):
        for kind in ("raw_input", "tray"):
            for failure in (
                None,
                "window_close",
                "create",
                "register_input",
                "message_loop",
                "callback",
                "device_removal",
                "power",
                "power_registration",
            ):
                if kind == "tray" and failure in {
                    "register_input",
                    "device_removal",
                    "power",
                    "power_registration",
                }:
                    continue
                with self.subTest(kind=kind, failure=failure):
                    registered = set()
                    windows = set()
                    power_handles = set()
                    user32 = Mock()
                    kernel32 = Mock()
                    kernel32.GetModuleHandleW.return_value = 1
                    callback = None

                    def register(pointer):
                        nonlocal callback
                        name = pointer._obj.lpszClassName
                        if name in registered:
                            return 0
                        registered.add(name)
                        callback = ctypes.CFUNCTYPE(
                            ctypes.wintypes.LPARAM,
                            ctypes.wintypes.HWND,
                            ctypes.wintypes.UINT,
                            ctypes.wintypes.WPARAM,
                            ctypes.wintypes.LPARAM,
                        )(pointer._obj.lpfnWndProc)
                        return 1

                    def create(*args):
                        if failure == "create":
                            return 0
                        windows.add(0x123456789)
                        return 0x123456789

                    def destroy(hwnd):
                        windows.remove(hwnd)
                        callback(hwnd, 0x0002, 0, 0)  # WM_DESTROY runs synchronously.
                        return 1

                    def unregister(name, instance):
                        self.assertFalse(windows)
                        self.assertFalse(power_handles)
                        registered.remove(name)
                        return 1

                    def register_power(hwnd, flags):
                        self.assertIn(hwnd, windows)
                        self.assertEqual(flags, 0)  # DEVICE_NOTIFY_WINDOW_HANDLE
                        if failure == "power_registration":
                            return 0
                        power_handles.add(0x987654321)
                        return 0x987654321

                    def unregister_power(handle):
                        power_handles.remove(handle)
                        return 1

                    user32.RegisterSuspendResumeNotification.side_effect = (
                        register_power
                    )
                    user32.UnregisterSuspendResumeNotification.side_effect = (
                        unregister_power
                    )

                    user32.RegisterClassW.side_effect = register
                    user32.CreateWindowExW.side_effect = create
                    user32.DestroyWindow.side_effect = destroy
                    user32.UnregisterClassW.side_effect = unregister
                    user32.RegisterRawInputDevices.return_value = (
                        0 if failure == "register_input" else 1
                    )
                    user32.RegisterWindowMessageW.return_value = 0xC000
                    user32.LoadIconW.return_value = 1

                    def get_message(*args):
                        if failure == "window_close":
                            callback(0x123456789, 0x0010, 0, 0)  # WM_CLOSE
                        if failure == "callback":
                            if kind == "raw_input":
                                source.publisher.device_removed.side_effect = OSError(
                                    "private callback failure"
                                )
                                callback(0x123456789, 0x00FE, 2, 42)
                            else:
                                source.status_provider.side_effect = OSError(
                                    "private callback failure"
                                )
                                callback(0x123456789, 0x0113, 0, 0)  # WM_TIMER
                        if failure == "device_removal":
                            callback(0x123456789, 0x00FE, 2, 42)
                        if failure == "power":
                            self.assertTrue(power_handles)
                            callback(0x123456789, 0x0218, 4, 0)
                        return -1 if failure == "message_loop" else 0

                    user32.GetMessageW.side_effect = get_message
                    with (
                        patch.object(sys, "platform", "win32"),
                        patch.object(
                            ctypes,
                            "windll",
                            SimpleNamespace(
                                user32=user32, kernel32=kernel32, shell32=Mock()
                            ),
                            create=True,
                        ),
                        patch.object(
                            ctypes, "WINFUNCTYPE", ctypes.CFUNCTYPE, create=True
                        ),
                        patch.object(
                            ctypes,
                            "WinError",
                            return_value=OSError("native failure"),
                            create=True,
                        ),
                    ):
                        source = (
                            RawInputPttSource(Mock(), Mock())
                            if kind == "raw_input"
                            else WindowsTrayIcon(
                                command_handler=Mock(),
                                status_provider=Mock(return_value="test"),
                            )
                        )
                        run = (
                            source._run_win32_loop
                            if kind == "raw_input"
                            else source.run
                        )
                        if failure == "callback" and kind == "raw_input":
                            run()
                            self.assertIsInstance(source.last_error, BackendError)
                            self.assertTrue(source._stop_requested.is_set())
                        elif failure not in (
                            None,
                            "window_close",
                            "device_removal",
                            "power",
                        ):
                            with self.assertRaises((OSError, BackendError)):
                                run()
                        else:
                            run()
                            run()
                        if failure == "device_removal":
                            self.assertEqual(
                                source.approval_publisher.device_removed.call_count, 2
                            )
                            source.approval_publisher.device_removed.assert_called_with(
                                42
                            )
                        if failure == "power":
                            self.assertEqual(
                                source.approval_publisher.clear.call_count, 2
                            )
                    self.assertIsNone(source._hwnd)
                    self.assertFalse(windows)
                    self.assertFalse(registered)
                    self.assertFalse(power_handles)
                    self.assertEqual(
                        user32.PostQuitMessage.call_count,
                        2
                        if failure == "window_close"
                        else 1
                        if failure == "callback"
                        else 0,
                    )


if __name__ == "__main__":
    unittest.main()
