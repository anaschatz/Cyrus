from __future__ import annotations

import json
import unittest
from datetime import UTC, datetime

import _bootstrap  # noqa: F401

from jarvis_orchestrator.domain import CanonicalTurnV1
from jarvis_orchestrator.runtime_context import (
    augment_system_prompt,
    capture_runtime_facts,
    resolve_temporal_query,
    select_relevant_history,
)


OBSERVED = datetime(2026, 8, 24, 17, 31, 42, tzinfo=UTC)


def turn(index: int, user: str, assistant: str = "answer") -> CanonicalTurnV1:
    return CanonicalTurnV1(
        turn_id=f"turn-{index}",
        user_text=user,
        user_language="en",
        assistant_text=assistant,
    )


class RuntimeContextTests(unittest.TestCase):
    def test_runtime_facts_are_bound_to_athens_at_turn_admission(self) -> None:
        facts = capture_runtime_facts(OBSERVED, timezone_name="Europe/Athens")

        self.assertEqual(facts.observed_at_utc, "2026-08-24T17:31:42.000Z")
        self.assertEqual(facts.local_date, "2026-08-24")
        self.assertEqual(facts.local_time, "20:31:42")
        self.assertEqual(facts.local_weekday, "Monday")
        self.assertEqual(facts.local_weekday_iso, 1)
        self.assertEqual(facts.utc_offset, "+03:00")

    def test_augmented_prompt_has_one_final_no_think_and_parseable_facts(self) -> None:
        facts = capture_runtime_facts(OBSERVED, timezone_name="Europe/Athens")
        prompt = augment_system_prompt("Answer directly. /no_think", facts)

        self.assertTrue(prompt.endswith("/no_think"))
        self.assertEqual(prompt.count("/no_think"), 1)
        encoded = prompt.split("Runtime facts JSON: ", 1)[1].rsplit(" /no_think", 1)[0]
        document = json.loads(encoded)
        self.assertEqual(document["local_date"], "2026-08-24")
        self.assertEqual(document["timezone"], "Europe/Athens")

    def test_closed_temporal_queries_are_bilingual_and_relative(self) -> None:
        today = resolve_temporal_query(
            "Τι μέρα είναι σήμερα;",
            input_language="el",
            response_language="match_input",
            observed_at=OBSERVED,
            timezone_name="Europe/Athens",
        )
        tomorrow = resolve_temporal_query(
            "What day is tomorrow?",
            input_language="en",
            response_language="match_input",
            observed_at=OBSERVED,
            timezone_name="Europe/Athens",
        )
        current_time = resolve_temporal_query(
            "Τι ώρα είναι τώρα;",
            input_language="el",
            response_language="en",
            observed_at=OBSERVED,
            timezone_name="Europe/Athens",
        )

        assert today is not None and tomorrow is not None and current_time is not None
        self.assertEqual(
            today.text,
            "Σήμερα είναι Δευτέρα, 24 Αυγούστου 2026.",
        )
        self.assertEqual(tomorrow.text, "Tomorrow is Tuesday, 25 August 2026.")
        self.assertEqual(current_time.text, "It is 20:31 in Europe/Athens.")

    def test_observed_asr_alias_is_accepted_only_for_greek_input(self) -> None:
        recovered = resolve_temporal_query(
            "The day is today",
            input_language="el",
            response_language="match_input",
            observed_at=OBSERVED,
            timezone_name="Europe/Athens",
        )
        ordinary_english = resolve_temporal_query(
            "The day is today",
            input_language="en",
            response_language="match_input",
            observed_at=OBSERVED,
            timezone_name="Europe/Athens",
        )

        assert recovered is not None
        self.assertIn("Δευτέρα", recovered.text)
        self.assertIsNone(ordinary_english)

    def test_history_selector_keeps_newest_plus_topic_relevance(self) -> None:
        history = (
            turn(1, "Explain random forests"),
            turn(2, "Explain transformer attention"),
            turn(3, "Give me pasta instructions"),
            turn(4, "How much VRAM does the model need?"),
        )

        selected = select_relevant_history("Compare transformer models", history)
        unrelated = select_relevant_history("Weather forecast", history)

        self.assertEqual([item.turn_id for item in selected], ["turn-2", "turn-4"])
        self.assertEqual([item.turn_id for item in unrelated], ["turn-4"])

    def test_new_explain_request_does_not_pull_unrelated_history(self) -> None:
        history = (
            turn(1, "Plan a trip to Japan"),
            turn(2, "Debug a Docker network"),
            turn(3, "Summarize a finance article"),
            turn(4, "How much VRAM does the model need?"),
        )

        selected = select_relevant_history(
            "Explain photosynthesis from first principles",
            history,
        )
        follow_up = select_relevant_history("Explain that in more detail", history)

        self.assertEqual([item.turn_id for item in selected], ["turn-4"])
        self.assertEqual(
            [item.turn_id for item in follow_up],
            ["turn-1", "turn-2", "turn-3", "turn-4"],
        )

    def test_naive_observation_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            capture_runtime_facts(
                datetime(2026, 8, 24, 17, 31),
                timezone_name="Europe/Athens",
            )
        with self.assertRaises(ValueError):
            resolve_temporal_query(
                "what time is it",
                input_language="en",
                response_language="match_input",
                observed_at=datetime(2026, 8, 24, 17, 31),
                timezone_name="Europe/Athens",
            )


if __name__ == "__main__":
    unittest.main()
