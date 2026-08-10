from __future__ import annotations

import json
import queue
import unittest

import _bootstrap  # noqa: F401

from jarvis_voice.ptt import (
    KEYBOARD_OVERRUN_MAKE_CODE,
    RI_KEY_BREAK,
    VK_F9,
    PttEvent,
    PttEventPublisher,
    PttTransition,
    RawKeyboardPacket,
    encode_raw_keyboard_payload,
    function_key_vkey,
    parse_raw_keyboard_payload,
)


DOWN = RawKeyboardPacket(0x43, 0, VK_F9)
UP = RawKeyboardPacket(0x43, RI_KEY_BREAK, VK_F9)


class PttTests(unittest.TestCase):
    def test_configurable_function_key_mapping(self) -> None:
        f12 = function_key_vkey("F12")
        publisher = PttEventPublisher(capacity=2, maximum_hold_seconds=1, ptt_vkey=f12)
        publisher.handle_packet("keyboard", RawKeyboardPacket(0x58, 0, f12), 1.0)
        self.assertEqual(publisher.events.get_nowait().transition, PttTransition.DOWN)
        publisher.handle_packet("keyboard", DOWN, 1.1)
        self.assertTrue(publisher.events.empty())

    def test_raw_keyboard_synthetic_buffer_round_trip(self) -> None:
        packet = RawKeyboardPacket(0x43, RI_KEY_BREAK, VK_F9, 0x101, 42)
        self.assertEqual(parse_raw_keyboard_payload(encode_raw_keyboard_payload(packet)), packet)

    def test_multiple_physical_keyboards_form_one_hold(self) -> None:
        publisher = PttEventPublisher(capacity=8, maximum_hold_seconds=10)
        publisher.handle_packet("keyboard-a", DOWN, 1.0)
        publisher.handle_packet("keyboard-a", DOWN, 1.1)
        publisher.handle_packet("keyboard-b", DOWN, 1.2)
        self.assertEqual(publisher.events.get_nowait().transition, PttTransition.DOWN)
        publisher.handle_packet("keyboard-a", UP, 2.0)
        with self.assertRaises(queue.Empty):
            publisher.events.get_nowait()
        publisher.handle_packet("keyboard-b", UP, 2.1)
        self.assertEqual(publisher.events.get_nowait().transition, PttTransition.UP)

    def test_overrun_and_other_keys_are_ignored(self) -> None:
        publisher = PttEventPublisher(capacity=4, maximum_hold_seconds=10)
        publisher.handle_packet("a", RawKeyboardPacket(KEYBOARD_OVERRUN_MAKE_CODE, 0, VK_F9), 1)
        publisher.handle_packet("a", RawKeyboardPacket(1, 0, 0x41), 1)
        self.assertEqual(publisher.held_device_count, 0)

    def test_device_removal_and_suspend_force_release(self) -> None:
        for action, reason in ((lambda p: p.device_removed("a", 2), "device_removed"), (lambda p: p.force_release(2, "suspend_resume"), "suspend_resume")):
            publisher = PttEventPublisher(capacity=4, maximum_hold_seconds=10)
            publisher.handle_packet("a", DOWN, 1); publisher.events.get_nowait()
            action(publisher)
            event = publisher.events.get_nowait()
            self.assertEqual(event.transition, PttTransition.FORCED_UP)
            self.assertEqual(event.reason, reason)

    def test_lost_break_watchdog_forces_release(self) -> None:
        publisher = PttEventPublisher(capacity=4, maximum_hold_seconds=3)
        publisher.handle_packet("a", DOWN, 10); publisher.events.get_nowait()
        self.assertFalse(publisher.watchdog(12.9))
        self.assertTrue(publisher.watchdog(13.0))
        self.assertEqual(publisher.events.get_nowait().reason, "watchdog")

    def test_overflow_lost_down_ignores_matching_up(self) -> None:
        publisher = PttEventPublisher(capacity=1, maximum_hold_seconds=10)
        publisher.events.put_nowait(PttEvent(PttTransition.DOWN, 0))
        publisher.handle_packet("secret-device", DOWN, 1)
        publisher.events.get_nowait()
        publisher.handle_packet("secret-device", UP, 2)
        with self.assertRaises(queue.Empty):
            publisher.events.get_nowait()
        self.assertNotIn("secret-device", json.dumps(publisher.errors[0].to_log_record()))

    def test_overflow_lost_up_sets_out_of_band_forced_release(self) -> None:
        publisher = PttEventPublisher(capacity=1, maximum_hold_seconds=10)
        publisher.handle_packet("a", DOWN, 1)
        publisher.handle_packet("a", UP, 2)
        forced = publisher.take_forced_release()
        self.assertIsNotNone(forced)
        self.assertEqual(forced.transition, PttTransition.FORCED_UP)
        self.assertEqual(publisher.errors[-1].code, "input.ptt_queue_overflow")

    def test_v1_is_passive_and_has_no_suppression_mode(self) -> None:
        publisher = PttEventPublisher(capacity=1, maximum_hold_seconds=1)
        self.assertTrue(publisher.passive)
        self.assertIsNone(publisher.suppression_mode)


if __name__ == "__main__":
    unittest.main()
