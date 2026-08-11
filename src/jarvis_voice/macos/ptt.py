"""Passive CoreGraphics event-tap PTT without third-party keyboard hooks."""

from __future__ import annotations

import ctypes
import sys
import threading
import time
from collections.abc import Callable
from typing import Protocol

from ..errors import BackendError
from ..ptt import ApprovalKeyPublisher, PttEventPublisher, RI_KEY_BREAK, RawKeyboardPacket, function_key_vkey

MAC_FUNCTION_KEYCODES = {
    "F1": 122,
    "F2": 120,
    "F3": 99,
    "F4": 118,
    "F5": 96,
    "F6": 97,
    "F7": 98,
    "F8": 100,
    "F9": 101,
    "F10": 109,
    "F11": 103,
    "F12": 111,
    "F13": 105,
    "F14": 107,
    "F15": 113,
    "F16": 106,
    "F17": 64,
    "F18": 79,
    "F19": 80,
    "F20": 90,
}

CG_EVENT_KEY_DOWN = 10
CG_EVENT_KEY_UP = 11
CG_EVENT_FLAGS_CHANGED = 12
CG_EVENT_TAP_DISABLED_BY_TIMEOUT = 0xFFFFFFFE
CG_EVENT_TAP_DISABLED_BY_USER_INPUT = 0xFFFFFFFF
CG_KEYBOARD_EVENT_AUTOREPEAT = 8
CG_KEYBOARD_EVENT_KEYCODE = 9
CG_EVENT_FLAG_MASK_CONTROL = 1 << 18
CG_EVENT_FLAG_MASK_ALTERNATE = 1 << 19

# Synthetic values are kept outside the native keycode/Win32 VK ranges. They
# exist only inside the passive macOS adapter and never leave the process.
MAC_CONTROL_OPTION_KEYCODE = -2
MAC_CONTROL_OPTION_VKEY = 0x100


def mac_function_keycode(name: str) -> int:
    try:
        return MAC_FUNCTION_KEYCODES[name.upper()]
    except (AttributeError, KeyError) as exc:
        raise BackendError(
            "config.unsupported_macos_ptt_key",
            "macOS PTT key must be F1 through F20",
            subsystem="config",
            safe_context={"field": "ptt_key"},
        ) from exc


def mac_ptt_keycode(name: str) -> int:
    if isinstance(name, str) and name.upper() == "CONTROL_OPTION":
        return MAC_CONTROL_OPTION_KEYCODE
    return mac_function_keycode(name)


def mac_ptt_vkey(name: str) -> int:
    if isinstance(name, str) and name.upper() == "CONTROL_OPTION":
        return MAC_CONTROL_OPTION_VKEY
    return function_key_vkey(name)


class EventTapAdapter(Protocol):
    def preflight(self) -> bool: ...
    def run(self, callback: Callable[[int, int, bool, float], None], on_ready: Callable[[], None]) -> None: ...
    def stop(self) -> None: ...


class CoreGraphicsEventTapAdapter:
    """ctypes wrapper kept behind a platform gate and a narrow test seam."""

    def __init__(self) -> None:
        self._run_loop: int | None = None
        self._core_foundation: ctypes.CDLL | None = None

    @staticmethod
    def _framework(name: str) -> ctypes.CDLL:
        return ctypes.CDLL(f"/System/Library/Frameworks/{name}.framework/{name}")

    def preflight(self) -> bool:
        if sys.platform != "darwin":
            return False
        cg = self._framework("ApplicationServices")
        check = getattr(cg, "CGPreflightListenEventAccess", None)
        if check is None:
            return True
        check.argtypes = []
        check.restype = ctypes.c_bool
        return bool(check())

    def run(self, callback: Callable[[int, int, bool, float], None], on_ready: Callable[[], None]) -> None:  # pragma: no cover - native TCC event loop
        cg = self._framework("ApplicationServices")
        cf = self._framework("CoreFoundation")
        self._core_foundation = cf
        event_callback_type = ctypes.CFUNCTYPE(
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.c_void_p,
            ctypes.c_void_p,
        )
        cg.CGEventGetIntegerValueField.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        cg.CGEventGetIntegerValueField.restype = ctypes.c_int64
        cg.CGEventGetFlags.argtypes = [ctypes.c_void_p]
        cg.CGEventGetFlags.restype = ctypes.c_uint64
        cg.CGEventTapCreate.argtypes = [
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint64,
            event_callback_type,
            ctypes.c_void_p,
        ]
        cg.CGEventTapCreate.restype = ctypes.c_void_p
        cg.CGEventTapEnable.argtypes = [ctypes.c_void_p, ctypes.c_bool]
        cf.CFMachPortCreateRunLoopSource.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_long]
        cf.CFMachPortCreateRunLoopSource.restype = ctypes.c_void_p
        cf.CFRunLoopGetCurrent.argtypes = []
        cf.CFRunLoopGetCurrent.restype = ctypes.c_void_p
        cf.CFRunLoopAddSource.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
        cf.CFRunLoopRun.argtypes = []
        cf.CFMachPortInvalidate.argtypes = [ctypes.c_void_p]
        cf.CFRelease.argtypes = [ctypes.c_void_p]

        modifier_chord_active = False

        def receive(_proxy: int, event_type: int, event: int, _context: int) -> int:
            nonlocal modifier_chord_active
            received_at = time.perf_counter()
            if event_type in {CG_EVENT_TAP_DISABLED_BY_TIMEOUT, CG_EVENT_TAP_DISABLED_BY_USER_INPUT}:
                callback(event_type, -1, False, received_at)
                cg.CGEventTapEnable(_proxy, True)
                return event
            if event_type == CG_EVENT_FLAGS_CHANGED:
                flags = int(cg.CGEventGetFlags(event))
                active = bool(flags & CG_EVENT_FLAG_MASK_CONTROL) and bool(
                    flags & CG_EVENT_FLAG_MASK_ALTERNATE
                )
                if active != modifier_chord_active:
                    modifier_chord_active = active
                    callback(
                        CG_EVENT_KEY_DOWN if active else CG_EVENT_KEY_UP,
                        MAC_CONTROL_OPTION_KEYCODE,
                        False,
                        received_at,
                    )
                return event
            keycode = int(cg.CGEventGetIntegerValueField(event, CG_KEYBOARD_EVENT_KEYCODE))
            autorepeat = bool(cg.CGEventGetIntegerValueField(event, CG_KEYBOARD_EVENT_AUTOREPEAT))
            callback(event_type, keycode, autorepeat, received_at)
            return event

        native_callback = event_callback_type(receive)
        mask = (1 << CG_EVENT_KEY_DOWN) | (1 << CG_EVENT_KEY_UP) | (1 << CG_EVENT_FLAGS_CHANGED)
        # kCGHIDEventTap=0, head insertion=0, listen-only=1.  Returning the
        # original event is an additional non-suppression guarantee.
        tap = cg.CGEventTapCreate(0, 0, 1, mask, native_callback, None)
        if not tap:
            raise BackendError("input.macos_event_tap_unavailable", "Passive event tap could not start", subsystem="input")
        source = cf.CFMachPortCreateRunLoopSource(None, tap, 0)
        if not source:
            cf.CFMachPortInvalidate(tap)
            cf.CFRelease(tap)
            raise BackendError("input.macos_event_tap_unavailable", "Passive event tap could not start", subsystem="input")
        run_loop = cf.CFRunLoopGetCurrent()
        self._run_loop = int(run_loop)
        common_modes = ctypes.c_void_p.in_dll(cf, "kCFRunLoopCommonModes").value
        cf.CFRunLoopAddSource(run_loop, source, common_modes)
        cg.CGEventTapEnable(tap, True)
        on_ready()
        try:
            cf.CFRunLoopRun()
        finally:
            self._run_loop = None
            cf.CFMachPortInvalidate(tap)
            cf.CFRelease(source)
            cf.CFRelease(tap)
            # Keep the callback alive until the tap and source are invalidated.
            del native_callback

    def stop(self) -> None:
        run_loop, cf = self._run_loop, self._core_foundation
        if run_loop is not None and cf is not None:
            cf.CFRunLoopStop.argtypes = [ctypes.c_void_p]
            cf.CFRunLoopStop(ctypes.c_void_p(run_loop))


class MacEventTapPttSource:
    passive = True
    suppression_mode = None

    def __init__(
        self,
        publisher: PttEventPublisher,
        *,
        key: str,
        adapter: EventTapAdapter | None = None,
        approval_publisher: ApprovalKeyPublisher | None = None,
    ) -> None:
        self.publisher = publisher
        self.keycode = mac_ptt_keycode(key)
        self.adapter = adapter or CoreGraphicsEventTapAdapter()
        self.approval_publisher = approval_publisher
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._error: Exception | None = None
        self._device = "mac_event_tap"

    def handle_event(self, event_type: int, keycode: int, autorepeat: bool, received_at: float) -> None:
        if event_type in {CG_EVENT_TAP_DISABLED_BY_TIMEOUT, CG_EVENT_TAP_DISABLED_BY_USER_INPUT}:
            self.publisher.force_release(received_at, "event_tap_disabled")
            return
        if event_type not in {CG_EVENT_KEY_DOWN, CG_EVENT_KEY_UP}:
            return
        if self.approval_publisher is not None and keycode in {MAC_FUNCTION_KEYCODES["F10"], MAC_FUNCTION_KEYCODES["F11"]}:
            flags = RI_KEY_BREAK if event_type == CG_EVENT_KEY_UP else 0
            self.approval_publisher.handle_packet(
                self._device,
                RawKeyboardPacket(
                    make_code=keycode,
                    flags=flags,
                    vkey=function_key_vkey("F10" if keycode == MAC_FUNCTION_KEYCODES["F10"] else "F11"),
                ),
                received_at,
                autorepeat=autorepeat,
            )
            return
        if keycode != self.keycode:
            return
        if event_type == CG_EVENT_KEY_DOWN and autorepeat:
            return
        flags = RI_KEY_BREAK if event_type == CG_EVENT_KEY_UP else 0
        self.publisher.handle_packet(
            self._device,
            RawKeyboardPacket(make_code=keycode, flags=flags, vkey=self.publisher.ptt_vkey),
            received_at,
        )

    def start(self) -> None:
        if sys.platform != "darwin":
            raise BackendError("input.macos_only", "CoreGraphics PTT requires macOS", subsystem="input")
        if self._thread is not None:
            return
        if not self.adapter.preflight():
            raise BackendError("input.macos_listen_permission_required", "Input Monitoring permission is required", subsystem="input")
        self._ready.clear()
        self._error = None
        self._thread = threading.Thread(target=self._run, name="jarvis-macos-event-tap", daemon=True)
        self._thread.start()
        self._ready.wait(timeout=5.0)
        if self._error is not None:
            self._thread = None
            if isinstance(self._error, BackendError):
                raise self._error
            raise BackendError("input.macos_event_tap_failed", "Passive event tap failed", subsystem="input") from self._error
        if not self._ready.is_set():
            self.adapter.stop()
            self._thread = None
            raise BackendError("input.macos_event_tap_timeout", "Passive event tap startup timed out", subsystem="input")

    def _run(self) -> None:
        try:
            self.adapter.run(self.handle_event, self._ready.set)
        except Exception as exc:
            self._error = exc
            self._ready.set()

    def stop(self) -> None:
        self.publisher.force_release(time.perf_counter(), "source_stop")
        self.adapter.stop()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        self._thread = None
