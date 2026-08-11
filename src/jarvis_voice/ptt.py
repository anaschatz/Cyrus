"""Physical-device PTT state and fail-safe bounded publication."""

from __future__ import annotations

import queue
import struct
from dataclasses import dataclass
from enum import Enum
from typing import Hashable

from .errors import BackendError

VK_F9 = 0x78
VK_F10 = 0x79
VK_F11 = 0x7A
VK_F1 = 0x70
RI_KEY_BREAK = 0x0001
KEYBOARD_OVERRUN_MAKE_CODE = 0x00FF
_RAWKEYBOARD = struct.Struct("<HHHHII")


class PttTransition(str, Enum):
    DOWN = "down"
    UP = "up"
    FORCED_UP = "forced_up"


@dataclass(frozen=True, slots=True)
class PttEvent:
    transition: PttTransition
    received_at: float
    reason: str = "raw_input"


@dataclass(frozen=True, slots=True)
class RawKeyboardPacket:
    make_code: int
    flags: int
    vkey: int
    message: int = 0
    extra_information: int = 0

    @property
    def is_break(self) -> bool:
        return bool(self.flags & RI_KEY_BREAK)


@dataclass(frozen=True, slots=True)
class ApprovalKeyEvent:
    key: str
    is_down: bool
    counter: int
    received_at: float
    autorepeat: bool = False


class ApprovalKeyPublisher:
    """Passive F10/F11 transitions with held-key and repeat suppression."""

    def __init__(self, *, capacity: int = 16) -> None:
        self.events: queue.Queue[ApprovalKeyEvent] = queue.Queue(maxsize=capacity)
        self._held: set[tuple[Hashable, str]] = set()
        self._counter = 0

    @property
    def counter(self) -> int:
        return self._counter

    @property
    def both_keys_up(self) -> bool:
        return not any(key in {"F10", "F11"} for _device, key in self._held)

    def handle_packet(
        self,
        device: Hashable,
        packet: RawKeyboardPacket,
        received_at: float,
        *,
        autorepeat: bool = False,
    ) -> None:
        key = "F10" if packet.vkey == VK_F10 else "F11" if packet.vkey == VK_F11 else None
        if key is None:
            return
        identity = (device, key)
        if packet.is_break:
            if identity not in self._held:
                return
            self._held.remove(identity)
            self._counter += 1
            self._publish(ApprovalKeyEvent(key, False, self._counter, received_at))
            return
        if autorepeat or identity in self._held:
            return
        self._held.add(identity)
        self._counter += 1
        self._publish(ApprovalKeyEvent(key, True, self._counter, received_at, autorepeat=False))

    def clear(self) -> None:
        self._held.clear()

    def _publish(self, event: ApprovalKeyEvent) -> None:
        try:
            self.events.put_nowait(event)
        except queue.Full as exc:
            raise BackendError(
                "input.approval_key_queue_overflow",
                "Approval key queue overflowed",
                subsystem="input",
            ) from exc


def parse_raw_keyboard_payload(payload: bytes) -> RawKeyboardPacket:
    """Parse the fixed RAWKEYBOARD payload used by synthetic buffer tests."""

    if len(payload) != _RAWKEYBOARD.size:
        raise BackendError("input.raw_keyboard_size", "RAWKEYBOARD payload has the wrong size", subsystem="input")
    make_code, flags, _reserved, vkey, message, extra = _RAWKEYBOARD.unpack(payload)
    return RawKeyboardPacket(make_code, flags, vkey, message, extra)


def encode_raw_keyboard_payload(packet: RawKeyboardPacket) -> bytes:
    return _RAWKEYBOARD.pack(packet.make_code, packet.flags, 0, packet.vkey, packet.message, packet.extra_information)


class PttEventPublisher:
    """Aggregate the configured function-key state across physical keyboards."""

    passive = True
    suppression_mode = None

    def __init__(self, *, capacity: int, maximum_hold_seconds: float, ptt_vkey: int = VK_F9) -> None:
        self.events: queue.Queue[PttEvent] = queue.Queue(maxsize=capacity)
        self.maximum_hold_seconds = maximum_hold_seconds
        self.ptt_vkey = ptt_vkey
        self._held_devices: set[Hashable] = set()
        self._down_received_at: float | None = None
        self._lost_down = False
        self._forced_release: PttEvent | None = None
        self.errors: list[BackendError] = []

    @property
    def held_device_count(self) -> int:
        return len(self._held_devices)

    def handle_packet(self, device: Hashable, packet: RawKeyboardPacket, received_at: float) -> None:
        if packet.make_code == KEYBOARD_OVERRUN_MAKE_CODE or packet.vkey != self.ptt_vkey:
            return
        if packet.is_break:
            self._release(device, received_at)
        else:
            self._press(device, received_at)

    def _press(self, device: Hashable, received_at: float) -> None:
        if device in self._held_devices:
            return
        was_idle = not self._held_devices
        self._held_devices.add(device)
        if not was_idle:
            return
        self._down_received_at = received_at
        if not self._publish(PttEvent(PttTransition.DOWN, received_at)):
            self._lost_down = True

    def _release(self, device: Hashable, received_at: float) -> None:
        if device not in self._held_devices:
            return
        self._held_devices.remove(device)
        if self._held_devices:
            return
        self._down_received_at = None
        if self._lost_down:
            self._lost_down = False
            return
        event = PttEvent(PttTransition.UP, received_at)
        if not self._publish(event):
            self._forced_release = PttEvent(PttTransition.FORCED_UP, received_at, "queue_overflow")

    def device_removed(self, device: Hashable, received_at: float) -> None:
        if device not in self._held_devices:
            return
        self._held_devices.remove(device)
        if not self._held_devices:
            self._force_after_held(received_at, "device_removed")

    def force_release(self, received_at: float, reason: str) -> None:
        if not self._held_devices:
            return
        self._held_devices.clear()
        self._force_after_held(received_at, reason)

    def watchdog(self, now: float) -> bool:
        if self._down_received_at is None or now - self._down_received_at < self.maximum_hold_seconds:
            return False
        self.force_release(now, "watchdog")
        return True

    def _force_after_held(self, received_at: float, reason: str) -> None:
        self._down_received_at = None
        if self._lost_down:
            self._lost_down = False
            return
        event = PttEvent(PttTransition.FORCED_UP, received_at, reason)
        if not self._publish(event):
            self._forced_release = event

    def take_forced_release(self) -> PttEvent | None:
        event, self._forced_release = self._forced_release, None
        return event

    def _publish(self, event: PttEvent) -> bool:
        try:
            self.events.put_nowait(event)
            return True
        except queue.Full:
            self.errors.append(
                BackendError(
                    "input.ptt_queue_overflow",
                    "PTT transition queue overflowed",
                    subsystem="input",
                    safe_context={"transition": event.transition.value},
                )
            )
            return False


def function_key_vkey(name: str) -> int:
    """Translate a validated F1..F24 configuration value to Win32 VK_* form."""

    normalized = name.upper()
    if not normalized.startswith("F") or not normalized[1:].isdigit():
        raise BackendError("config.unsupported_ptt_key", "PTT key must be F1 through F24", subsystem="config", safe_context={"field": "ptt_key"})
    number = int(normalized[1:])
    if not 1 <= number <= 24:
        raise BackendError("config.unsupported_ptt_key", "PTT key must be F1 through F24", subsystem="config", safe_context={"field": "ptt_key"})
    return VK_F1 + number - 1
