"""Native Shell_NotifyIcon tray with a minimal, non-semantic command surface."""

from __future__ import annotations

import ctypes
import sys
from ctypes import wintypes
from enum import IntEnum
from typing import Callable, Protocol

from ..errors import BackendError

NIM_ADD = 0x00000000
NIM_MODIFY = 0x00000001
NIM_DELETE = 0x00000002
NIM_SETVERSION = 0x00000004
NOTIFYICON_VERSION_4 = 4
NIF_MESSAGE = 0x00000001
NIF_ICON = 0x00000002
NIF_TIP = 0x00000004
WM_DESTROY = 0x0002
WM_CLOSE = 0x0010
WM_QUERYENDSESSION = 0x0011
WM_ENDSESSION = 0x0016
WM_COMMAND = 0x0111
WM_CONTEXTMENU = 0x007B
WM_LBUTTONUP = 0x0202
WM_RBUTTONUP = 0x0205
WM_TIMER = 0x0113
WM_APP = 0x8000
TRAY_CALLBACK = WM_APP + 1
MF_STRING = 0x0000
MF_SEPARATOR = 0x0800
MF_GRAYED = 0x0001
TPM_RIGHTBUTTON = 0x0002
TPM_RETURNCMD = 0x0100
IDI_APPLICATION = 32512


class TrayCommand(IntEnum):
    START = 1001
    STOP = 1002
    OPEN_CONFIG = 1003
    OPEN_LOGS = 1004
    EXIT = 1005


class TrayShell(Protocol):
    def add(self, tooltip: str) -> bool: ...
    def set_version(self) -> bool: ...
    def modify(self, tooltip: str) -> bool: ...
    def delete(self) -> None: ...


class TrayCoordinator:
    """Fake-testable recovery and command routing around the native shell."""

    def __init__(self, shell: TrayShell, *, command_handler: Callable[[TrayCommand], None], status_provider: Callable[[], str]) -> None:
        self.shell = shell
        self.command_handler = command_handler
        self.status_provider = status_provider
        self.active = False

    def start(self) -> None:
        if not self._register():
            raise BackendError("tray.icon_add_failed", "The notification icon could not be registered", subsystem="tray")

    def _register(self) -> bool:
        if not self.shell.add(self.status_provider()):
            self.active = False
            return False
        if not self.shell.set_version():
            self.shell.delete()
            self.active = False
            return False
        self.active = True
        return True

    def refresh(self) -> None:
        if not self.active:
            self._register()
        elif not self.shell.modify(self.status_provider()):
            self.shell.delete()
            self.active = False

    def taskbar_created(self) -> None:
        self.active = False
        self._register()

    def dispatch(self, command: int) -> None:
        try:
            selected = TrayCommand(command)
        except ValueError:
            return
        self.command_handler(selected)

    def close(self) -> None:
        if self.active:
            self.shell.delete()
            self.active = False


class _GUID(ctypes.Structure):
    _fields_ = [("Data1", wintypes.DWORD), ("Data2", wintypes.WORD), ("Data3", wintypes.WORD), ("Data4", ctypes.c_ubyte * 8)]


class _NOTIFYICONDATAW(ctypes.Structure):
    _fields_ = [
        ("cbSize", wintypes.DWORD), ("hWnd", wintypes.HWND), ("uID", wintypes.UINT),
        ("uFlags", wintypes.UINT), ("uCallbackMessage", wintypes.UINT), ("hIcon", wintypes.HICON),
        ("szTip", wintypes.WCHAR * 128), ("dwState", wintypes.DWORD), ("dwStateMask", wintypes.DWORD),
        ("szInfo", wintypes.WCHAR * 256), ("uVersion", wintypes.UINT), ("szInfoTitle", wintypes.WCHAR * 64),
        ("dwInfoFlags", wintypes.DWORD), ("guidItem", _GUID), ("hBalloonIcon", wintypes.HICON),
    ]


class Win32TrayShell:
    def __init__(self, hwnd: int, icon: int) -> None:
        self.hwnd = hwnd
        self.icon = icon
        self.shell32 = ctypes.windll.shell32
        self.shell32.Shell_NotifyIconW.argtypes = [wintypes.DWORD, ctypes.POINTER(_NOTIFYICONDATAW)]
        self.shell32.Shell_NotifyIconW.restype = wintypes.BOOL

    def _data(self, tooltip: str = "") -> _NOTIFYICONDATAW:
        value = _NOTIFYICONDATAW()
        value.cbSize = ctypes.sizeof(_NOTIFYICONDATAW)
        value.hWnd = self.hwnd
        value.uID = 1
        value.uFlags = NIF_MESSAGE | NIF_ICON | NIF_TIP
        value.uCallbackMessage = TRAY_CALLBACK
        value.hIcon = self.icon
        value.szTip = tooltip[:127]
        return value

    def add(self, tooltip: str) -> bool:
        value = self._data(tooltip)
        return bool(self.shell32.Shell_NotifyIconW(NIM_ADD, ctypes.byref(value)))

    def set_version(self) -> bool:
        value = self._data()
        value.uVersion = NOTIFYICON_VERSION_4
        return bool(self.shell32.Shell_NotifyIconW(NIM_SETVERSION, ctypes.byref(value)))

    def modify(self, tooltip: str) -> bool:
        value = self._data(tooltip)
        return bool(self.shell32.Shell_NotifyIconW(NIM_MODIFY, ctypes.byref(value)))

    def delete(self) -> None:
        value = self._data()
        self.shell32.Shell_NotifyIconW(NIM_DELETE, ctypes.byref(value))


class WindowsTrayIcon:
    def __init__(self, *, command_handler: Callable[[TrayCommand], None], status_provider: Callable[[], str]) -> None:
        if sys.platform != "win32":
            raise BackendError("tray.windows_only", "The native tray requires Windows", subsystem="tray")
        self.command_handler = command_handler
        self.status_provider = status_provider
        self._hwnd: int | None = None

    def request_close(self) -> None:  # pragma: no cover - native Windows only
        if self._hwnd is not None:
            post_message = ctypes.windll.user32.PostMessageW
            post_message.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
            post_message.restype = wintypes.BOOL
            post_message(self._hwnd, WM_CLOSE, 0, 0)

    def run(self) -> None:  # pragma: no cover - native Windows only
        user32, kernel32 = ctypes.windll.user32, ctypes.windll.kernel32
        LRESULT = wintypes.LPARAM
        WNDPROC = ctypes.WINFUNCTYPE(LRESULT, wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM)

        class WNDCLASSW(ctypes.Structure):
            _fields_ = [("style", wintypes.UINT), ("lpfnWndProc", ctypes.c_void_p), ("cbClsExtra", ctypes.c_int), ("cbWndExtra", ctypes.c_int), ("hInstance", wintypes.HINSTANCE), ("hIcon", wintypes.HICON), ("hCursor", wintypes.HANDLE), ("hbrBackground", wintypes.HBRUSH), ("lpszMenuName", wintypes.LPCWSTR), ("lpszClassName", wintypes.LPCWSTR)]

        kernel32.GetModuleHandleW.argtypes = [wintypes.LPCWSTR]
        kernel32.GetModuleHandleW.restype = wintypes.HMODULE
        user32.RegisterWindowMessageW.argtypes = [wintypes.LPCWSTR]
        user32.RegisterWindowMessageW.restype = wintypes.UINT
        user32.RegisterClassW.argtypes = [ctypes.POINTER(WNDCLASSW)]
        user32.RegisterClassW.restype = wintypes.ATOM
        user32.CreateWindowExW.argtypes = [wintypes.DWORD, wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.DWORD, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, wintypes.HWND, wintypes.HMENU, wintypes.HINSTANCE, wintypes.LPVOID]
        user32.CreateWindowExW.restype = wintypes.HWND
        user32.DestroyWindow.argtypes = [wintypes.HWND]
        user32.DestroyWindow.restype = wintypes.BOOL
        user32.CreatePopupMenu.argtypes = []
        user32.CreatePopupMenu.restype = wintypes.HMENU
        user32.AppendMenuW.argtypes = [wintypes.HMENU, wintypes.UINT, wintypes.WPARAM, wintypes.LPCWSTR]
        user32.AppendMenuW.restype = wintypes.BOOL
        user32.TrackPopupMenu.argtypes = [wintypes.HMENU, wintypes.UINT, ctypes.c_int, ctypes.c_int, ctypes.c_int, wintypes.HWND, wintypes.LPVOID]
        user32.TrackPopupMenu.restype = wintypes.UINT
        user32.DestroyMenu.argtypes = [wintypes.HMENU]
        user32.DestroyMenu.restype = wintypes.BOOL
        user32.GetCursorPos.argtypes = [ctypes.POINTER(wintypes.POINT)]
        user32.GetCursorPos.restype = wintypes.BOOL
        user32.SetForegroundWindow.argtypes = [wintypes.HWND]
        user32.SetForegroundWindow.restype = wintypes.BOOL
        user32.SetTimer.argtypes = [wintypes.HWND, wintypes.WPARAM, wintypes.UINT, wintypes.LPVOID]
        user32.SetTimer.restype = wintypes.WPARAM
        user32.LoadIconW.argtypes = [wintypes.HINSTANCE, wintypes.LPCWSTR]
        user32.LoadIconW.restype = wintypes.HICON
        user32.GetMessageW.argtypes = [ctypes.POINTER(wintypes.MSG), wintypes.HWND, wintypes.UINT, wintypes.UINT]
        user32.GetMessageW.restype = ctypes.c_int
        user32.DefWindowProcW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
        user32.DefWindowProcW.restype = LRESULT
        taskbar_created = user32.RegisterWindowMessageW("TaskbarCreated")
        if not taskbar_created:
            raise BackendError("tray.taskbar_message_failed", "Taskbar recovery message registration failed", subsystem="tray")
        coordinator: TrayCoordinator | None = None

        def show_menu(hwnd: int) -> None:
            menu = user32.CreatePopupMenu()
            status = self.status_provider()[:96]
            user32.AppendMenuW(menu, MF_STRING | MF_GRAYED, 0, status)
            user32.AppendMenuW(menu, MF_SEPARATOR, 0, None)
            user32.AppendMenuW(menu, MF_STRING, TrayCommand.START, "Start Voice")
            user32.AppendMenuW(menu, MF_STRING, TrayCommand.STOP, "Stop Voice")
            user32.AppendMenuW(menu, MF_SEPARATOR, 0, None)
            user32.AppendMenuW(menu, MF_STRING, TrayCommand.OPEN_CONFIG, "Open Config Folder")
            user32.AppendMenuW(menu, MF_STRING, TrayCommand.OPEN_LOGS, "Open Log Folder")
            user32.AppendMenuW(menu, MF_SEPARATOR, 0, None)
            user32.AppendMenuW(menu, MF_STRING, TrayCommand.EXIT, "Exit")
            point = wintypes.POINT()
            user32.GetCursorPos(ctypes.byref(point))
            user32.SetForegroundWindow(hwnd)
            command = user32.TrackPopupMenu(menu, TPM_RIGHTBUTTON | TPM_RETURNCMD, point.x, point.y, 0, hwnd, None)
            user32.DestroyMenu(menu)
            if command and coordinator is not None:
                coordinator.dispatch(command)

        def window_proc(hwnd: int, message: int, wparam: int, lparam: int) -> int:
            if message == taskbar_created and coordinator is not None:
                coordinator.taskbar_created(); return 0
            if message == TRAY_CALLBACK:
                notification = lparam & 0xFFFF
                if notification in {WM_CONTEXTMENU, WM_RBUTTONUP, WM_LBUTTONUP}:
                    show_menu(hwnd)
                return 0
            if message == WM_TIMER and coordinator is not None:
                coordinator.refresh(); return 0
            if message == WM_COMMAND and coordinator is not None:
                coordinator.dispatch(wparam & 0xFFFF); return 0
            if message == WM_QUERYENDSESSION:
                return 1
            if message == WM_ENDSESSION and wparam:
                if coordinator is not None:
                    coordinator.dispatch(TrayCommand.EXIT)
                return 0
            if message == WM_CLOSE:
                user32.DestroyWindow(hwnd); return 0
            if message == WM_DESTROY:
                if coordinator is not None:
                    coordinator.close()
                user32.PostQuitMessage(0); return 0
            return user32.DefWindowProcW(hwnd, message, wparam, lparam)

        callback = WNDPROC(window_proc)
        instance = kernel32.GetModuleHandleW(None)
        class_name = "JarvisVoiceTrayWindow"
        window_class = WNDCLASSW(0, ctypes.cast(callback, ctypes.c_void_p), 0, 0, instance, None, None, None, None, class_name)
        if not user32.RegisterClassW(ctypes.byref(window_class)):
            raise BackendError("tray.window_class_failed", "Tray window class registration failed", subsystem="tray")
        hwnd = user32.CreateWindowExW(0, class_name, class_name, 0, 0, 0, 0, 0, None, None, instance, None)
        if not hwnd:
            raise BackendError("tray.window_create_failed", "Tray window creation failed", subsystem="tray")
        self._hwnd = int(hwnd)
        icon_resource = ctypes.cast(ctypes.c_void_p(IDI_APPLICATION), wintypes.LPCWSTR)
        icon = user32.LoadIconW(None, icon_resource)
        coordinator = TrayCoordinator(Win32TrayShell(self._hwnd, int(icon)), command_handler=self.command_handler, status_provider=self.status_provider)
        try:
            coordinator.start()
            user32.SetTimer(hwnd, 1, 500, None)
            message = wintypes.MSG()
            while True:
                result = user32.GetMessageW(ctypes.byref(message), None, 0, 0)
                if result == -1:
                    raise BackendError("tray.message_loop_failed", "Tray message loop failed", subsystem="tray")
                if result == 0:
                    break
                user32.TranslateMessage(ctypes.byref(message)); user32.DispatchMessageW(ctypes.byref(message))
        finally:
            coordinator.close()
            self._hwnd = None
