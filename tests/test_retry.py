from __future__ import annotations

import unittest

import _bootstrap  # noqa: F401

from jarvis_voice.fakes import FakeClock
from jarvis_voice.protocol import new_uuid4
from jarvis_voice.retry import AckRetryScheduler


class RetryTests(unittest.TestCase):
    def test_exact_retry_schedule_and_deadlines_from_initial_send(self) -> None:
        clock = FakeClock(100.0)
        scheduler = AckRetryScheduler(new_uuid4(), clock)
        scheduler.start()
        self.assertEqual(scheduler.due_attempts(), [0])
        clock.advance(0.49); self.assertEqual(scheduler.due_attempts(), [])
        clock.advance(0.01); self.assertEqual(scheduler.due_attempts(), [1])
        clock.advance(1.0); self.assertEqual(scheduler.due_attempts(), [2])
        clock.advance(2.0); self.assertEqual(scheduler.due_attempts(), [3])
        clock.advance(0.49); self.assertFalse(scheduler.connected_timed_out)
        clock.advance(0.01); self.assertTrue(scheduler.connected_timed_out)
        clock.advance(26.0); self.assertTrue(scheduler.reconnect_ttl_expired)

    def test_ack_stops_retries_and_only_matches_original_event(self) -> None:
        event_id = new_uuid4(); clock = FakeClock(); scheduler = AckRetryScheduler(event_id, clock); scheduler.start()
        self.assertFalse(scheduler.acknowledge(new_uuid4()))
        self.assertTrue(scheduler.acknowledge(event_id))
        clock.advance(100)
        self.assertEqual(scheduler.due_attempts(), [])
        self.assertFalse(scheduler.connected_timed_out)


if __name__ == "__main__":
    unittest.main()

