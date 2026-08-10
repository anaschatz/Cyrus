from __future__ import annotations

import unittest

from jarvis_inference.circuit import RestartCircuitBreaker


class CircuitTests(unittest.TestCase):
    def test_fourth_restart_is_blocked_then_one_half_open_probe(self) -> None:
        breaker = RestartCircuitBreaker(limit=3, window=300, cooldown=300)
        for instant in (0.0, 1.0, 2.0):
            self.assertTrue(breaker.allow(instant))
            breaker.record_restart(instant)
        self.assertFalse(breaker.allow(3.0))
        self.assertFalse(breaker.allow(302.9))
        self.assertTrue(breaker.allow(303.0))
        self.assertFalse(breaker.allow(303.1))
        breaker.probe_succeeded()
        self.assertTrue(breaker.allow(303.2))

    def test_failed_half_open_probe_reopens_without_sticking_inflight(self) -> None:
        breaker = RestartCircuitBreaker(limit=1, window=10, cooldown=10)
        self.assertTrue(breaker.allow(0)); breaker.record_restart(0)
        self.assertFalse(breaker.allow(1))
        self.assertTrue(breaker.allow(11))
        breaker.record_restart(11)
        self.assertFalse(breaker.allow(12))
        self.assertTrue(breaker.allow(21))
