from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from jarvis_inference.profile_preflight import Fixture
from jarvis_inference.profile_quality import FROZEN_CORPUS_SHA256, load_corpus, run_quality_gate


CORPUS = Path(__file__).parents[1] / "fixtures/local-fallback-quality-v1.json"


class FakeClient:
    def __init__(self, failed: set[str] | None = None, unsafe: set[str] | None = None):
        self.failed = failed or set()
        self.unsafe = unsafe or set()

    def readiness(self) -> None:
        pass

    def capabilities(self) -> None:
        pass

    def stream(self, fixture: Fixture) -> dict:
        passed = fixture.case_id not in self.failed
        return {
            "case_id": fixture.case_id,
            "language": fixture.language,
            "category": fixture.category,
            "protocol_pass": True,
            "language_ok": True,
            "exact_ok": passed,
            "required_content_ok": True,
            "forbidden_content_ok": fixture.case_id not in self.unsafe,
            "word_limit_ok": True,
            "completed_ok": True,
            "quality_pass": passed,
            "finish_reason": "stop",
            "ttft_seconds": 0.1,
            "end_to_end_seconds": 0.5,
            "completion_tokens": 10,
            "decode_tokens_per_second": 30.0,
            "output_utf8_bytes": 8,
        }


class ProfileQualityTests(unittest.TestCase):
    def test_frozen_corpus_is_exactly_balanced(self) -> None:
        fixtures, digest = load_corpus(CORPUS)
        self.assertEqual(digest, FROZEN_CORPUS_SHA256)
        self.assertEqual(len(fixtures), 80)
        self.assertEqual(sum(item.language == "el" for item in fixtures), 40)
        self.assertEqual(sum(item.language == "en" for item in fixtures), 40)

    def test_corpus_digest_mismatch_fails_closed(self) -> None:
        raw = CORPUS.read_bytes().replace(b"jarvis-local-fallback-quality-v1", b"changed-quality-suite", 1)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "corpus.json"
            path.write_bytes(raw)
            with self.assertRaises(RuntimeError):
                load_corpus(path)

    def test_all_cases_pass_without_emitting_content(self) -> None:
        fixtures, digest = load_corpus(CORPUS)
        report = run_quality_gate(FakeClient(), fixtures, digest, warmups=1)  # type: ignore[arg-type]
        self.assertEqual(report["status"], "deterministic_quality_pass")
        self.assertEqual(report["sample_count"], 80)
        self.assertNotIn("text", str(report).lower())

    def test_language_threshold_and_critical_content_are_fail_closed(self) -> None:
        fixtures, digest = load_corpus(CORPUS)
        greek_failures = {f"el_instruction_{index:02d}" for index in range(1, 8)}
        report = run_quality_gate(FakeClient(greek_failures), fixtures, digest, warmups=1)  # type: ignore[arg-type]
        self.assertEqual(report["language_results"]["el"]["passed"], 33)
        self.assertEqual(report["status"], "deterministic_quality_failed")
        unsafe = run_quality_gate(FakeClient(unsafe={"en_knowledge_01"}), fixtures, digest, warmups=1)  # type: ignore[arg-type]
        self.assertEqual(unsafe["critical_failures"], 1)
        self.assertFalse(unsafe["thresholds"]["critical_failures_zero"])


if __name__ == "__main__":
    unittest.main()
