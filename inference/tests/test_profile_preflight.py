from __future__ import annotations

import unittest

from jarvis_inference.profile_preflight import FIXTURES, Fixture, fixture_digest, percentile, run_preflight, score_output


class ProfilePreflightTests(unittest.TestCase):
    def test_fixture_identity_is_stable_and_complete(self) -> None:
        self.assertEqual(len(FIXTURES), 6)
        self.assertEqual({item.language for item in FIXTURES}, {"el", "en"})
        self.assertRegex(fixture_digest(), r"^[0-9a-f]{64}$")

    def test_exact_and_summary_scoring(self) -> None:
        exact = Fixture("exact", "el", "prompt", 8, exact="Αθήνα")
        self.assertTrue(score_output(exact, "Αθήνα", "stop")["quality_pass"])
        self.assertFalse(score_output(exact, "Αθήνα.", "stop")["quality_pass"])
        numeric = Fixture("numeric", "el", "prompt", 8, exact="42")
        self.assertTrue(score_output(numeric, "42", "stop")["quality_pass"])
        boundary = Fixture("boundary", "el", "prompt", 8, exact="PRIMARY_REQUIRED")
        self.assertTrue(score_output(boundary, "PRIMARY_REQUIRED", "stop")["quality_pass"])
        summary = Fixture(
            "summary", "en", "prompt", 16,
            required_groups=(("bread",), ("milk",)), max_words=5,
        )
        self.assertTrue(score_output(summary, "John bought bread and milk", "stop")["quality_pass"])
        self.assertFalse(score_output(summary, "John bought bread and milk at the market", "stop")["quality_pass"])

    def test_length_finish_and_missing_content_fail(self) -> None:
        fixture = Fixture(
            "explain", "en", "prompt", 32,
            required_groups=(("blue",), ("scatter",), ("air", "atmosphere")),
        )
        self.assertTrue(score_output(fixture, "Blue light scatters in the atmosphere.", "stop")["quality_pass"])
        self.assertFalse(score_output(fixture, "Blue light scatters in the atmosphere", "length")["quality_pass"])
        self.assertFalse(score_output(fixture, "The sky is blue.", "stop")["quality_pass"])

    def test_percentile_is_interpolated_and_rejects_empty_input(self) -> None:
        self.assertEqual(percentile([1.0, 2.0, 3.0], 0.5), 2.0)
        self.assertAlmostEqual(percentile([1.0, 2.0], 0.95), 1.95)
        with self.assertRaises(ValueError):
            percentile([], 0.95)

    def test_report_is_sanitized_and_non_authoritative(self) -> None:
        class Client:
            def readiness(self) -> None:
                pass

            def capabilities(self) -> None:
                pass

            def stream(self, fixture: Fixture) -> dict:
                return {
                    "case_id": fixture.case_id,
                    "language": fixture.language,
                    "protocol_pass": True,
                    "language_ok": True,
                    "exact_ok": True,
                    "required_content_ok": True,
                    "forbidden_content_ok": True,
                    "word_limit_ok": True,
                    "completed_ok": True,
                    "quality_pass": True,
                    "finish_reason": "stop",
                    "ttft_seconds": 0.1,
                    "end_to_end_seconds": 0.5,
                    "completion_tokens": 10,
                    "decode_tokens_per_second": 30.0,
                    "output_utf8_bytes": 8,
                }

        report = run_preflight(Client(), warmups=1, repetitions=1)  # type: ignore[arg-type]
        self.assertEqual(report["status"], "preflight_pass")
        self.assertFalse(report["authoritative_qualification"])
        self.assertNotIn("text", str(report).lower())


if __name__ == "__main__":
    unittest.main()
