from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from jarvis_inference.errors import ApplianceError
from jarvis_inference.durable import read_report
from jarvis_inference.manifest import load_manifest
from jarvis_inference.qualification import SUITE_VERSION, evaluate_evidence, qualify
from jarvis_inference.strict_json import canonical_bytes


def artifact(path: Path, root: Path) -> dict:
    raw = path.read_bytes()
    return {"path": path.relative_to(root).as_posix(), "size_bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


class QualificationTests(unittest.TestCase):
    @staticmethod
    def identity() -> dict:
        return {"id": "exact", "hardware_runtime": {"driver": "test", "compute_runtime": "test"}}

    def manifest(self, root: Path):
        exe = root / "server.exe"; exe.write_bytes(b"x")
        model = root / "model.gguf"; model.write_bytes(b"x")
        value = {
            "manifest_schema_version": 1, "alias": "jarvis-local", "platform": "windows-x86_64",
            "backend": {"type": "llama-server", "llama_cpp_commit": "a" * 40, "build_id": "x", "executable": artifact(exe, root), "dlls": [], "parameters": {}},
            "model": {"format": "gguf", "context_window": 4096, "artifacts": [artifact(model, root)]},
            "python": {}, "capabilities": {"text": True, "stream": True, "tools": False, "json_object": False, "json_schema": False},
            "qualification_report": "report.json",
        }
        path = root / "manifest.json"; path.write_text(json.dumps(value))
        return load_manifest(path)

    def provenance(self) -> dict:
        return {
            "challenge": "c" * 64, "identity_digest": "d" * 64,
            "not_before_unix_ns": 100, "not_after_unix_ns": 300,
            "max_elapsed_ns": 500,
        }

    def evidence(self, identity: dict) -> dict:
        power = {
            "windows_power_scheme_guid": "12345678-1234-1234-1234-123456789abc",
            "ac_state": "ac", "gpu_power_limit_watts": 360,
            "gpu_clock_policy": "fixed", "driver_version": "test",
            "compute_runtime": "test", "require_zero_thermal_power_throttle_reasons": True,
        }
        power_digest = hashlib.sha256(canonical_bytes(power)).hexdigest()
        return {
            "suite_version": SUITE_VERSION, "fixture_digest": "b" * 64,
            "deterministic_seeds": [1], "identity": identity,
            "provenance": {
                "challenge": "c" * 64, "identity_digest": "d" * 64,
                "fixture_digest": "b" * 64, "started_unix_ns": "101",
                "completed_unix_ns": "299", "monotonic_elapsed_ns": 400,
            },
            "streaming": {"greek_success": 20, "english_success": 20},
            "tools": {"named_success": 0, "required_success": 0},
            "json_schema": {"strict_success": 0},
            "cancellation": {"prefill_success": 5, "mid_stream_success": 5},
            "crash_restart": {"isolated_success": 5},
            "isolation": {"pairs_success": 50, "leaks": 0},
            "protocol_conformance": 1.0, "privacy_leaks": 0, "unexpected_oom": 0,
            "unexpected_crashes": 0, "unexpected_hangs": 0,
            "platform_gate": {
                "kind": "windows_asr_coexistence",
                "fixture": {
                    "rendered_prompt_token_lengths": [512, 2048, 4096],
                    "max_completion_tokens": 512, "request_count": 100, "concurrency": 1,
                    "schedule": "serial-continuous-from-audio-start", "minimum_overlapping_utterances": 80,
                    "sampling": {"temperature": 0, "top_p": 1}, "power_profile": power,
                },
                "audio_corpus_digest": "e" * 64, "greek_utterances": 50, "english_utterances": 50,
                "baseline_power_identity_digest": power_digest, "coexistence_power_identity_digest": power_digest,
                "overlapping_utterances": 80, "baseline_p95": 1.0, "coexistence_p95": 1.15,
                "lost_final_transcripts": 0, "oom": 0, "crashes": 0, "throttle_reasons": 0,
            },
        }

    def test_numeric_gates_are_exact_and_bounded(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest = self.manifest(Path(temporary))
            identity = self.identity()
            evidence = self.evidence(identity)
            summary = evaluate_evidence(manifest, evidence, identity, self.provenance())
            self.assertEqual(summary["fixture_digest"], "b" * 64)
            evidence["platform_gate"]["coexistence_p95"] = 1.151
            with self.assertRaises(ApplianceError):
                evaluate_evidence(manifest, evidence, identity, self.provenance())

    def test_static_or_mismatched_evidence_cannot_pass(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest = self.manifest(Path(temporary))
            identity = self.identity()
            evidence = self.evidence(identity)
            evidence["provenance"]["challenge"] = "old"
            with self.assertRaises(ApplianceError) as caught:
                evaluate_evidence(manifest, evidence, identity, self.provenance())
            self.assertEqual(caught.exception.code, "qualification_provenance_mismatch")

    def test_failure_durably_replaces_in_progress_state(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); manifest = self.manifest(root)
            failure = ApplianceError(503, "injected_native_failure", "sanitized")
            with patch("jarvis_inference.qualification.assert_runtime_platform", side_effect=failure):
                with self.assertRaises(ApplianceError):
                    qualify(manifest, lambda _: self.fail("engine must not spawn"), root / "runner")
            report = read_report(manifest.qualification_report)
            self.assertIsNotNone(report)
            self.assertEqual(report["status"], "failed")
            self.assertEqual(report["failure_code"], "injected_native_failure")
            self.assertNotIn("challenge", report)

    def test_nan_and_boolean_metrics_cannot_satisfy_gates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest = self.manifest(Path(temporary)); identity = self.identity()
            for field, value in (("protocol_conformance", True), ("protocol_conformance", float("nan"))):
                evidence = self.evidence(identity); evidence[field] = value
                with self.subTest(value=value), self.assertRaises(ApplianceError):
                    evaluate_evidence(manifest, evidence, identity, self.provenance())
