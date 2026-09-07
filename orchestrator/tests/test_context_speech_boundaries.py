from __future__ import annotations

import unittest
from datetime import UTC, datetime, tzinfo

import _bootstrap  # noqa: F401
from jarvis_orchestrator.runtime_context import (
    capture_runtime_facts,
    resolve_temporal_query,
    select_relevant_history,
)
from jarvis_orchestrator.speech import SpeechSegmenterV1
from test_runtime_context import turn


class ContextSpeechBoundaryTests(unittest.TestCase):
    def test_history_limit_is_exact_and_integral(self):
        history = (turn(1, "transformer models"), turn(2, "pasta recipes"))
        self.assertEqual(
            select_relevant_history("transformer", history, maximum_pairs=1),
            (history[-1],),
        )
        for maximum in (True, 1.5, float("nan")):
            with self.subTest(maximum=maximum), self.assertRaises(ValueError):
                select_relevant_history("transformer", history, maximum_pairs=maximum)

    def test_speech_bound_is_a_positive_integer_before_any_input(self):
        for maximum in (0, -1, True, 1.5, float("nan")):
            with self.subTest(maximum=maximum), self.assertRaises(ValueError):
                SpeechSegmenterV1(maximum_characters=maximum)

    def test_canonically_equivalent_greek_stops_segment_the_same(self):
        for stop in ("\u0387", "\u00b7"):
            with self.subTest(stop=stop):
                split = SpeechSegmenterV1()
                self.assertEqual(split.feed(f"Πρώτο{stop} Δεύτερο."), ("Πρώτο·",))
                self.assertEqual(split.finish(), ("Δεύτερο.",))

    def test_offsetless_tzinfo_is_rejected_instead_of_using_host_timezone(self):
        class Offsetless(tzinfo):
            def utcoffset(self, dt):
                return None

        observed = datetime(2026, 1, 1, tzinfo=Offsetless())
        with self.assertRaises(ValueError):
            capture_runtime_facts(observed, timezone_name="UTC")
        with self.assertRaises(ValueError):
            resolve_temporal_query(
                "what time is it",
                input_language="en",
                response_language="en",
                observed_at=observed,
                timezone_name="UTC",
            )

    def test_temporal_query_outside_datetime_range_fails_closed(self):
        for query, observed in (
            ("what day is tomorrow", datetime.max.replace(tzinfo=UTC)),
            ("what day was yesterday", datetime.min.replace(tzinfo=UTC)),
        ):
            with self.subTest(query=query):
                self.assertIsNone(
                    resolve_temporal_query(
                        query,
                        input_language="en",
                        response_language="en",
                        observed_at=observed,
                        timezone_name="UTC",
                    )
                )


if __name__ == "__main__":
    unittest.main()
