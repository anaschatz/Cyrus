"""Passive Win32 Raw Input source using a message-only window."""

from __future__ import annotations

import sys
import threading
import time
from typing import Any

from ..errors import BackendError
from ..ptt import ApprovalKeyPublisher, PttEventPublisher, RawKeyboardPacket

WM_INPUT = 0x00FF
WM_INPUT_DEVICE_CHANGE = 0x00FE
WM_POWERBROADCAST = 0x0218
WM_CLOSE = 0x0010
WM_DESTROY = 0x0002
RIM_TYPEKEYBOARD = 1
RID_INPUT = 0x10000003
RIDEV_INPUTSINK = 0x00000100
RIDEV_DEVNOTIFY = 0x00002000
GIDC_REMOVAL = 2
PBT_APMSUSPEND = 0x0004
PBT_APMRESUMESUSPEND = 0x0007
PBT_APMRESUMEAUTOMATIC = 0x0012


class RawInputPttSource:
    """Windows-only source. Raw Input observes F9 and never suppresses it."""

    passive = True

    def __init__(self, publisher: PttEventPublisher, approval_publisher: ApprovalKeyPublisher | None = None) -> None:
        self.publisher = publisher
        self.approval_publisher = approval_publisher
        self._thread: threading.Thread | None = None
        self._hwnd: int | None = None
        self._ready = threading.Event()
        self._stop_requested = threading.Event()
        self._error: BackendError | None = None

    def start(self) -> None:
        if sys.platform != "win32":
            raise BackendError("input.raw_input_windows_only", "Raw Input requires Windows", subsystem="input")
        if self._thread is not None and self._thread.is_alive():
            if self._stop_requested.is_set():
                raise BackendError("input.raw_input_stopping", "Raw Input is still stopping", subsystem="input")
            return
        self._ready.clear()
        self._stop_requested.clear()
        self._error = None
        self._thread = threading.Thread(target=self._message_loop, name="jarvis-raw-input", daemon=True)
        self._thread.start()
        self._ready.wait(timeout=5)
        if self._error is not None:
            raise BackendError("input.raw_input_start_failed", "Raw Input could not start", subsystem="input") from self._error
        if self._hwnd is None:
            self._stop_requested.set()
            raise BackendError("input.raw_input_start_timeout", "Raw Input startup timed out", subsystem="input")

    def stop(self) -> None:
        self._stop_requested.set()
        if sys.platform != "win32":
            return
        if self._hwnd is not None:
            import ctypes
            from ctypes import wintypes
            post_message = ctypes.windll.user32.PostMessageW
            post_message.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
            post_message.restype = wintypes.BOOL
            post_message(self._hwnd, WM_CLOSE, 0, 0)
        if self._thread is not None:
            self._thread.join(timeout=5)
            if self._thread.is_alive():
                return
        self._thread = None
        self._hwnd = None

    @property
    def last_error(self) -> BackendError | None:
        return self._error

    def _fail(self, error: Exception) -> None:
        if self._error is None:
            if isinstance(error, BackendError):
                self._error = error
            else:
                self._error = BackendError("input.raw_input_failed", "Raw Input failed", subsystem="input")
                self._error.__cause__ = error
        self._stop_requested.set()
        self._ready.set()

    def _message_loop(self) -> None:  # pragma: no cover - native Windows only
        try:
            self._run_win32_loop()
            if not self._stop_requested.is_set():
                self._fail(BackendError("input.raw_input_closed", "Raw Input stopped unexpectedly", subsystem="input"))
        except Exception as exc:
            self._fail(exc)
        finally:
            self._ready.set()

    def _run_win32_loop(self) -> None:  # pragma: no cover - native Windows only
        import ctypes
        from ctypes import wintypes

        user32 = ctypes.windll.user32
        kernel32 = ctypes.windll.kernel32
        ULONG_PTR = wintypes.WPARAM
        LRESULT = wintypes.LPARAM

        class WNDCLASSW(ctypes.Structure):
            _fields_ = [("style", wintypes.UINT), ("lpfnWndProc", ctypes.c_void_p), ("cbClsExtra", ctypes.c_int), ("cbWndExtra", ctypes.c_int), ("hInstance", wintypes.HINSTANCE), ("hIcon", wintypes.HICON), ("hCursor", wintypes.HANDLE), ("hbrBackground", wintypes.HBRUSH), ("lpszMenuName", wintypes.LPCWSTR), ("lpszClassName", wintypes.LPCWSTR)]

        class RAWINPUTDEVICE(ctypes.Structure):
            _fields_ = [("usUsagePage", wintypes.USHORT), ("usUsage", wintypes.USHORT), ("dwFlags", wintypes.DWORD), ("hwndTarget", wintypes.HWND)]

        class RAWINPUTHEADER(ctypes.Structure):
            _fields_ = [("dwType", wintypes.DWORD), ("dwSize", wintypes.DWORD), ("hDevice", wintypes.HANDLE), ("wParam", ULONG_PTR)]

        class RAWKEYBOARD(ctypes.Structure):
            _fields_ = [("MakeCode", wintypes.USHORT), ("Flags", wintypes.USHORT), ("Reserved", wintypes.USHORT), ("VKey", wintypes.USHORT), ("Message", wintypes.UINT), ("ExtraInformation", wintypes.ULONG)]

        class RAWINPUT(ctypes.Structure):
            _fields_ = [("header", RAWINPUTHEADER), ("keyboard", RAWKEYBOARD)]

        WNDPROC = ctypes.WINFUNCTYPE(LRESULT, wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM)

        # ctypes otherwise assumes a C ``int`` result.  That silently truncates
        # pointer-sized HWND/HMODULE values in a 64-bit process.
        kernel32.GetModuleHandleW.argtypes = [wintypes.LPCWSTR]
        kernel32.GetModuleHandleW.restype = wintypes.HMODULE
        user32.RegisterClassW.argtypes = [ctypes.POINTER(WNDCLASSW)]
        user32.RegisterClassW.restype = wintypes.ATOM
        user32.UnregisterClassW.argtypes = [wintypes.LPCWSTR, wintypes.HINSTANCE]
        user32.UnregisterClassW.restype = wintypes.BOOL
        user32.RegisterSuspendResumeNotification.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        user32.RegisterSuspendResumeNotification.restype = wintypes.HANDLE
        user32.UnregisterSuspendResumeNotification.argtypes = [wintypes.HANDLE]
        user32.UnregisterSuspendResumeNotification.restype = wintypes.BOOL
        user32.CreateWindowExW.argtypes = [
            wintypes.DWORD,
            wintypes.LPCWSTR,
            wintypes.LPCWSTR,
            wintypes.DWORD,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            wintypes.HWND,
            wintypes.HMENU,
            wintypes.HINSTANCE,
            wintypes.LPVOID,
        ]
        user32.CreateWindowExW.restype = wintypes.HWND
        user32.RegisterRawInputDevices.argtypes = [ctypes.POINTER(RAWINPUTDEVICE), wintypes.UINT, wintypes.UINT]
        user32.RegisterRawInputDevices.restype = wintypes.BOOL
        user32.GetRawInputData.argtypes = [wintypes.HANDLE, wintypes.UINT, wintypes.LPVOID, ctypes.POINTER(wintypes.UINT), wintypes.UINT]
        user32.GetRawInputData.restype = wintypes.UINT
        user32.DefWindowProcW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
        user32.DefWindowProcW.restype = LRESULT
        user32.DestroyWindow.argtypes = [wintypes.HWND]
        user32.DestroyWindow.restype = wintypes.BOOL
        user32.GetMessageW.argtypes = [ctypes.POINTER(wintypes.MSG), wintypes.HWND, wintypes.UINT, wintypes.UINT]
        user32.GetMessageW.restype = ctypes.c_int

        window_destroyed = False
        pumping = False

        def dispatch_message(hwnd: Any, message: int, wparam: int, lparam: int) -> int:
            nonlocal window_destroyed
            received_at = time.perf_counter()
            if message == WM_INPUT:
                size = wintypes.UINT(0)
                header_size = ctypes.sizeof(RAWINPUTHEADER)
                user32.GetRawInputData(lparam, RID_INPUT, None, ctypes.byref(size), header_size)
                if size.value:
                    buffer = ctypes.create_string_buffer(size.value)
                    if user32.GetRawInputData(lparam, RID_INPUT, buffer, ctypes.byref(size), header_size) == size.value:
                        raw = ctypes.cast(buffer, ctypes.POINTER(RAWINPUT)).contents
                        if raw.header.dwType == RIM_TYPEKEYBOARD:
                            keyboard = raw.keyboard
                            device_id = int(raw.header.hDevice or 0)
                            packet = RawKeyboardPacket(keyboard.MakeCode, keyboard.Flags, keyboard.VKey, keyboard.Message, keyboard.ExtraInformation)
                            self.publisher.handle_packet(device_id, packet, received_at)
                            if self.approval_publisher is not None:
                                self.approval_publisher.handle_packet(device_id, packet, received_at)
                return 0
            if message == WM_INPUT_DEVICE_CHANGE and wparam == GIDC_REMOVAL:
                self.publisher.device_removed(int(lparam), received_at)
                if self.approval_publisher is not None:
                    self.approval_publisher.device_removed(int(lparam))
                return 0
            if message == WM_POWERBROADCAST and wparam in {
                PBT_APMSUSPEND,
                PBT_APMRESUMESUSPEND,
                PBT_APMRESUMEAUTOMATIC,
            }:
                self.publisher.force_release(received_at, "suspend_resume")
                if self.approval_publisher is not None:
                    self.approval_publisher.clear()
                return 1
            if message == WM_CLOSE:
                user32.DestroyWindow(hwnd); return 0
            if message == WM_DESTROY:
                window_destroyed = True
                if pumping:
                    user32.PostQuitMessage(0)
                return 0
            return user32.DefWindowProcW(hwnd, message, wparam, lparam)

        def window_proc(hwnd: Any, message: int, wparam: int, lparam: int) -> int:
            try:
                return dispatch_message(hwnd, message, wparam, lparam)
            except Exception as exc:
                self._fail(exc)
                if pumping:
                    user32.PostQuitMessage(0)
                return 0

        callback = WNDPROC(window_proc)
        class_name = "JarvisVoiceRawInputWindow"
        instance = kernel32.GetModuleHandleW(None)
        window_class = WNDCLASSW(0, ctypes.cast(callback, ctypes.c_void_p), 0, 0, instance, None, None, None, None, class_name)
        atom = user32.RegisterClassW(ctypes.byref(window_class))
        if not atom:
            raise ctypes.WinError()
        hwnd = None
        power_notification = None
        try:
            hwnd_message = wintypes.HWND(-3)
            hwnd = user32.CreateWindowExW(0, class_name, class_name, 0, 0, 0, 0, 0, hwnd_message, None, instance, None)
            if not hwnd:
                raise ctypes.WinError()
            power_notification = user32.RegisterSuspendResumeNotification(hwnd, 0)
            if not power_notification:
                raise ctypes.WinError()
            device = RAWINPUTDEVICE(0x01, 0x06, RIDEV_INPUTSINK | RIDEV_DEVNOTIFY, hwnd)
            if not user32.RegisterRawInputDevices(ctypes.byref(device), 1, ctypes.sizeof(device)):
                raise ctypes.WinError()
            self._hwnd = int(hwnd); self._ready.set()
            message = wintypes.MSG()
            pumping = True
            while not self._stop_requested.is_set():
                result = user32.GetMessageW(ctypes.byref(message), None, 0, 0)
                if result == -1:
                    raise ctypes.WinError()
                if result == 0:
                    break
                user32.TranslateMessage(ctypes.byref(message)); user32.DispatchMessageW(ctypes.byref(message))
        finally:
            pumping = False
            self._hwnd = None
            try:
                if power_notification:
                    user32.UnregisterSuspendResumeNotification(power_notification)
            finally:
                try:
                    if hwnd and not window_destroyed:
                        user32.DestroyWindow(hwnd)
                finally:
                    user32.UnregisterClassW(class_name, instance)
