from __future__ import annotations

import tempfile
import unittest
from collections import Counter
from pathlib import Path

from jarvis_inference.primary_profile import (
    PRIMARY_CORPUS_SHA256,
    load_primary_corpus,
    run_primary_quality_gate,
)
from jarvis_inference.profile_preflight import PreflightFailure


CORPUS = Path(__file__).parents[1] / "fixtures" / "local-primary-quality-v1.json"


class _PassingClient:
    def readiness(self) -> None:
        pass

    def capabilities(self) -> None:
        pass

    def stream(self, fixture):
        return {
            "case_id": fixture.case_id,
            "language": fixture.language,
            "category": fixture.category,
            "protocol_pass": True,
            "language_ok": True,
            "exact_ok": True,
            "required_content_ok": True,
            "forbidden_content_ok": True,
            "word_limit_ok": True,
            "completed_ok": True,
            "quality_pass": True,
            "finish_reason": "stop",
            "ttft_seconds": 0.2,
            "end_to_end_seconds": 1.0,
            "completion_tokens": 8,
            "decode_tokens_per_second": 40.0,
            "output_utf8_bytes": 8,
        }


class PrimaryProfileTests(unittest.TestCase):
    def test_frozen_composite_corpus_has_sixty_cases_per_language(self) -> None:
        fixtures, digest = load_primary_corpus(CORPUS)
        self.assertEqual(digest, PRIMARY_CORPUS_SHA256)
        self.assertEqual(len(fixtures), 120)
        self.assertEqual(Counter(item.language for item in fixtures), {"el": 60, "en": 60})
        self.assertEqual(
            set(Counter((item.language, item.category) for item in fixtures).values()),
            {12},
        )
        self.assertEqual(len({item.case_id for item in fixtures}), 120)

    def test_tampered_descriptor_fails_before_any_request(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "local-primary-quality-v1.json").write_bytes(
                CORPUS.read_bytes().replace(b"CX-4821", b"CX-4822", 1)
            )
            with self.assertRaisesRegex(PreflightFailure, "digest mismatch"):
                load_primary_corpus(root / "local-primary-quality-v1.json")

    def test_all_passing_results_satisfy_primary_thresholds(self) -> None:
        fixtures, digest = load_primary_corpus(CORPUS)
        report = run_primary_quality_gate(
            _PassingClient(), fixtures, digest, warmups=1
        )
        self.assertEqual(report["status"], "deterministic_quality_pass")
        self.assertEqual(report["language_results"]["el"]["passed"], 60)
        self.assertEqual(report["language_results"]["en"]["passed"], 60)
        self.assertTrue(all(report["thresholds"].values()))
        for case in report["cases"]:
            self.assertNotIn("text", case)
            self.assertNotIn("raw_output", case)


if __name__ == "__main__":
    unittest.main()
