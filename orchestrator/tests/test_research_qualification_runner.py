from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

import _bootstrap  # noqa: F401

from jarvis_orchestrator.domain import canonical_json
from jarvis_research.qualification import (
    MANDATORY_LIVE_CASES,
    LiveQualificationRunnerV1,
    _compose_command,
    main,
    qualification_digest,
    record_digest,
    verify_accepted_evidence,
    _run,
)

ROOT = Path(__file__).resolve().parents[2]


class ResearchQualificationRunnerTests(unittest.TestCase):
    def test_manifest_requires_every_mandatory_live_case_exactly_once(self) -> None:
        manifest = json.loads((ROOT / "research" / "qualification.manifest.template.json").read_text())
        manifest["cases"] = []
        with self.assertRaisesRegex(ValueError, "mandatory"):
            LiveQualificationRunnerV1(manifest)
        manifest["cases"] = [
            {
                "case_id": case_id,
                "idempotency_key": "q-" + case_id,
                "query": case_id,
                "expected_http": 200,
                "expected_source_count": 0,
                "cancel_after_ms": 0,
            }
            for case_id in sorted(MANDATORY_LIVE_CASES)
        ]
        LiveQualificationRunnerV1(manifest)

    def test_checked_in_template_is_strict_complete_and_model_free(self) -> None:
        manifest = json.loads((ROOT / "research" / "qualification.manifest.template.json").read_text())
        LiveQualificationRunnerV1(manifest)
        self.assertEqual({item["case_id"] for item in manifest["cases"]}, MANDATORY_LIVE_CASES)
        serialized = canonical_json(manifest)
        self.assertNotIn("model", serialized.casefold())
        self.assertFalse(manifest["request_template"]["plan"]["browser_fallback_allowed"])

    def test_compose_discovery_is_bounded_to_cli_or_bundled_plugin(self) -> None:
        command = _compose_command()
        self.assertIn(len(command), {1, 2})
        self.assertTrue(command[0].endswith(("docker", "docker-compose")))

    def test_subprocess_environment_keeps_only_bounded_docker_inputs(self) -> None:
        captured = {}
        original = subprocess.run

        def fake_run(command, **kwargs):
            captured.update(kwargs["env"])
            return subprocess.CompletedProcess(command, 0, "", "")

        try:
            subprocess.run = fake_run
            os.environ["JARVIS_RESEARCH_IMAGE"] = "sha256:" + "a" * 64
            os.environ["HTTP_PROXY"] = "http://forbidden.invalid"
            _run(["python3", "--version"], cwd=ROOT)
        finally:
            subprocess.run = original
            os.environ.pop("JARVIS_RESEARCH_IMAGE", None)
            os.environ.pop("HTTP_PROXY", None)
        self.assertIn("JARVIS_RESEARCH_IMAGE", captured)
        self.assertNotIn("HTTP_PROXY", captured)

    def test_runner_failure_is_canonical_rejected_evidence_with_digest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "manifest.json"
            output = root / "evidence.json"
            manifest.write_text("{}", encoding="utf-8")
            self.assertEqual(main(["--manifest", str(manifest), "--output", str(output)]), 2)
            evidence = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(evidence["gate_status"], "REJECTED")
            self.assertFalse(evidence["browser_enabled"])
            self.assertEqual(evidence["record_sha256"], record_digest(evidence))

    def test_accepted_evidence_digest_is_non_circular_and_tamper_evident(self) -> None:
        identity = {
            "authority_registry_version": "authority-registry/1",
            "browser_qualification_sha256": "",
            "contract_version": "research-corpus/1",
            "language_detector_sha256": "a" * 64,
            "language_detector_version": "jarvis-language-script/1",
            "model_sha256": "b" * 64,
            "profile_sha256": "c" * 64,
            "qualification_sha256": "0" * 64,
        }
        case = {
            "case_id": "authenticated_api", "outcome": "PASS", "code": "ok",
            "duration_ms": 12, "response_sha256": "d" * 64,
        }
        evidence = {
            "schema_version": 2, "gate_status": "ACCEPTED", "browser_enabled": False,
            "qualification_identity": identity, "artifact_sha256": {"lock": "e" * 64},
            "image_sha256": {"research-api": "f" * 64}, "cases": [case],
        }
        digest = qualification_digest(evidence)
        identity["qualification_sha256"] = digest
        evidence["cases"].append({
            "case_id": "digest_binding", "outcome": "PASS", "code": "ok",
            "duration_ms": 0, "response_sha256": "",
        })
        evidence["evidence_sha256"] = digest
        evidence["record_sha256"] = record_digest(evidence)
        verify_accepted_evidence(evidence)

        evidence["cases"][0]["duration_ms"] = 99
        self.assertEqual(qualification_digest(evidence), digest)
        with self.assertRaisesRegex(ValueError, "record digest"):
            verify_accepted_evidence(evidence)
        evidence["record_sha256"] = record_digest(evidence)
        evidence["artifact_sha256"]["lock"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "digest binding"):
            verify_accepted_evidence(evidence)


if __name__ == "__main__":
    unittest.main()
