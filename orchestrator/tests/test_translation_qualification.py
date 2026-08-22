from __future__ import annotations

import asyncio
import hashlib
import json
import os
import runpy
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import _bootstrap  # noqa: F401


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = (
    REPOSITORY_ROOT
    / "orchestrator"
    / "scripts"
    / "run_translation_qualification.py"
)
SCRIPT = runpy.run_path(str(SCRIPT_PATH))
CORPUS = (
    REPOSITORY_ROOT
    / "orchestrator"
    / "eval"
    / "greek-english-voice-v1.jsonl"
)
HOLDOUT = (
    REPOSITORY_ROOT
    / "orchestrator"
    / "eval"
    / "greek-english-voice-v1-holdout.jsonl"
)
EVIDENCE = (
    REPOSITORY_ROOT
    / "orchestrator"
    / "results"
    / "translation-qualification-m3-v1.json"
)


class TranslationQualificationTests(unittest.TestCase):
    def test_tracked_corpus_is_closed_and_substantial(self) -> None:
        fixtures = SCRIPT["load_fixtures"](CORPUS)

        self.assertEqual(len(fixtures), 82)
        self.assertEqual(len({item["id"] for item in fixtures}), 82)
        self.assertTrue(all(item["source"].strip() for item in fixtures))
        self.assertEqual(
            hashlib.sha256(CORPUS.read_bytes()).hexdigest(),
            "2a528e62c340a3614d3e4f948244b37ebccfaa832ef23b110d4f25d3c4c2593e",
        )
        holdout = SCRIPT["load_fixtures"](HOLDOUT)
        self.assertEqual(len(holdout), 25)
        self.assertEqual(len({item["id"] for item in holdout}), 25)
        self.assertEqual(
            hashlib.sha256(HOLDOUT.read_bytes()).hexdigest(),
            "c22e1be6ebb8ad2b87ab32aa2d263c23348c7a12b40edbb364bf6521684ede28",
        )

    def test_fixture_checks_exact_meaning_and_protected_spans(self) -> None:
        fixture = {
            "id": "example",
            "source": "Δοκιμή",
            "exact": "Keep F9 unchanged.",
            "required_all": ["keep"],
            "required_any": [["unchanged", "intact"]],
            "preserve": ["F9"],
        }
        SCRIPT["_validate_fixture"](fixture, "Keep F9 unchanged.")

        for output in (
            "Keep F8 unchanged.",
            "Discard F9 unchanged.",
            "Keep F9 modified.",
        ):
            with self.subTest(output=output):
                with self.assertRaises(SCRIPT["QualificationFailure"]):
                    SCRIPT["_validate_fixture"](fixture, output)

        structured = {
            "id": "structured",
            "source": "Δοκιμή",
            "required_all": ["no"],
            "forbidden": ["delete"],
            "ordered": ["first", "then", "stop"],
        }
        SCRIPT["_validate_fixture"](
            structured,
            "No: first check, then stop.",
        )
        for output in (
            "Nothing: first check, then stop.",
            "No: first delete, then stop.",
            "No: stop, then check first.",
        ):
            with self.subTest(output=output):
                with self.assertRaises(SCRIPT["QualificationFailure"]):
                    SCRIPT["_validate_fixture"](structured, output)

    def test_report_is_private_and_never_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            SCRIPT["_write_report"](path, {"status": "passed"})

            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            with self.assertRaises(SCRIPT["QualificationFailure"]):
                SCRIPT["_write_report"](path, {"status": "changed"})
            self.assertEqual(path.read_text(), '{"status":"passed"}')

    def test_tracked_evidence_is_bound_to_current_qualified_code(self) -> None:
        evidence = json.loads(EVIDENCE.read_text())
        worker = (
            REPOSITORY_ROOT
            / "orchestrator"
            / "src"
            / "jarvis_orchestrator"
            / "translation_worker.py"
        )

        self.assertEqual(evidence["status"], "passed")
        self.assertEqual(evidence["network_policy"], "external_deny_network")
        self.assertEqual(
            evidence["qualification_script_sha256"],
            hashlib.sha256(SCRIPT_PATH.read_bytes()).hexdigest(),
        )
        self.assertEqual(
            evidence["worker_runtime"]["worker_module_sha256"],
            hashlib.sha256(worker.read_bytes()).hexdigest(),
        )
        self.assertEqual(
            hashlib.sha256(SCRIPT["canonical_json"](evidence).encode()).hexdigest(),
            "6dd119e622b6cdb3aae51c945d96735f814755a8713acf5e20e9ea1c60d5198f",
        )

    def test_qualification_requires_external_network_denial_attestation(self) -> None:
        with self.assertRaises(SCRIPT["QualificationFailure"]) as raised:
            asyncio.run(
                SCRIPT["qualify"](
                    model_root=Path("/missing/model"),
                    manifest_path=Path("/missing/manifest"),
                    manifest_sha256="0" * 64,
                    build_lock_path=Path("/missing/build-lock"),
                    corpus_path=CORPUS,
                    holdout_path=HOLDOUT,
                    worker_python=Path("/missing/python"),
                    network_denied=False,
                )
            )

        self.assertEqual(
            raised.exception.code,
            "translation_qualification_network_policy_required",
        )

    @unittest.skipUnless(os.name == "posix", "POSIX process liveness contract")
    def test_missing_pid_is_not_reported_alive(self) -> None:
        self.assertFalse(SCRIPT["_pid_is_alive"](999_999_999))

    @unittest.skipUnless(
        SCRIPT["platform"].system() == "Darwin",
        "native macOS RSS contract",
    )
    def test_darwin_rss_probe_does_not_require_ps(self) -> None:
        with mock.patch.object(
            SCRIPT["subprocess"],
            "run",
            side_effect=PermissionError,
        ):
            rss_kib = SCRIPT["_rss_kib"](os.getpid())

        self.assertIsInstance(rss_kib, int)
        self.assertGreater(rss_kib, 0)


if __name__ == "__main__":
    unittest.main()
