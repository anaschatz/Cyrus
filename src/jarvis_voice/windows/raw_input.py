"""Passive Win32 Raw Input source using a message-only window."""

from __future__ import annotations

import sys
import threading
from typing import Any

from ..errors import BackendError
from ..ptt import PttEventPublisher, RawKeyboardPacket

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

    def __init__(self, publisher: PttEventPublisher) -> None:
        self.publisher = publisher
        self._thread: threading.Thread | None = None
        self._hwnd: int | None = None
        self._ready = threading.Event()
        self._error: Exception | None = None

    def start(self) -> None:
        if sys.platform != "win32":
            raise BackendError("input.raw_input_windows_only", "Raw Input requires Windows", subsystem="input")
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._message_loop, name="jarvis-raw-input", daemon=True)
        self._thread.start()
        self._ready.wait(timeout=5)
        if self._error is not None:
            raise BackendError("input.raw_input_start_failed", "Raw Input could not start", subsystem="input") from self._error
        if self._hwnd is None:
            raise BackendError("input.raw_input_start_timeout", "Raw Input startup timed out", subsystem="input")

    def stop(self) -> None:
        if self._hwnd is None or sys.platform != "win32":
            return
        import ctypes
        from ctypes import wintypes
        post_message = ctypes.windll.user32.PostMessageW
        post_message.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
        post_message.restype = wintypes.BOOL
        post_message(self._hwnd, WM_CLOSE, 0, 0)
        if self._thread is not None:
            self._thread.join(timeout=5)
        self._thread = None
        self._hwnd = None

    def _message_loop(self) -> None:  # pragma: no cover - native Windows only
        try:
            self._run_win32_loop()
        except Exception as exc:
            self._error = exc
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
        kernel32.QueryPerformanceCounter.argtypes = [ctypes.POINTER(ctypes.c_longlong)]
        kernel32.QueryPerformanceCounter.restype = wintypes.BOOL
        kernel32.QueryPerformanceFrequency.argtypes = [ctypes.POINTER(ctypes.c_longlong)]
        kernel32.QueryPerformanceFrequency.restype = wintypes.BOOL
        user32.RegisterClassW.argtypes = [ctypes.POINTER(WNDCLASSW)]
        user32.RegisterClassW.restype = wintypes.ATOM
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

        def qpc() -> float:
            counter, frequency = ctypes.c_longlong(), ctypes.c_longlong()
            if not kernel32.QueryPerformanceCounter(ctypes.byref(counter)):
                raise ctypes.WinError()
            if not kernel32.QueryPerformanceFrequency(ctypes.byref(frequency)):
                raise ctypes.WinError()
            return counter.value / frequency.value

        def window_proc(hwnd: Any, message: int, wparam: int, lparam: int) -> int:
            received_at = qpc()
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
                            self.publisher.handle_packet(int(raw.header.hDevice or 0), RawKeyboardPacket(keyboard.MakeCode, keyboard.Flags, keyboard.VKey, keyboard.Message, keyboard.ExtraInformation), received_at)
                return 0
            if message == WM_INPUT_DEVICE_CHANGE and wparam == GIDC_REMOVAL:
                self.publisher.device_removed(int(lparam), received_at); return 0
            if message == WM_POWERBROADCAST and wparam in {
                PBT_APMSUSPEND,
                PBT_APMRESUMESUSPEND,
                PBT_APMRESUMEAUTOMATIC,
            }:
                self.publisher.force_release(received_at, "suspend_resume"); return 1
            if message == WM_CLOSE:
                user32.DestroyWindow(hwnd); return 0
            if message == WM_DESTROY:
                user32.PostQuitMessage(0); return 0
            return user32.DefWindowProcW(hwnd, message, wparam, lparam)

        callback = WNDPROC(window_proc)
        class_name = "JarvisVoiceRawInputWindow"
        instance = kernel32.GetModuleHandleW(None)
        window_class = WNDCLASSW(0, ctypes.cast(callback, ctypes.c_void_p), 0, 0, instance, None, None, None, None, class_name)
        atom = user32.RegisterClassW(ctypes.byref(window_class))
        if not atom:
            raise ctypes.WinError()
        hwnd_message = wintypes.HWND(-3)
        hwnd = user32.CreateWindowExW(0, class_name, class_name, 0, 0, 0, 0, 0, hwnd_message, None, instance, None)
        if not hwnd:
            raise ctypes.WinError()
        device = RAWINPUTDEVICE(0x01, 0x06, RIDEV_INPUTSINK | RIDEV_DEVNOTIFY, hwnd)
        if not user32.RegisterRawInputDevices(ctypes.byref(device), 1, ctypes.sizeof(device)):
            raise ctypes.WinError()
        self._hwnd = int(hwnd); self._ready.set()
        message = wintypes.MSG()
        while True:
            result = user32.GetMessageW(ctypes.byref(message), None, 0, 0)
            if result == -1:
                raise ctypes.WinError()
            if result == 0:
                break
            user32.TranslateMessage(ctypes.byref(message)); user32.DispatchMessageW(ctypes.byref(message))
