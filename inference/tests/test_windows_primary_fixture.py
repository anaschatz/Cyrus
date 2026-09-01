from __future__ import annotations

import copy
import hashlib
import tempfile
import unittest
from collections import Counter
from decimal import Decimal
from pathlib import Path

from jarvis_inference.contracts import Capabilities, normalize_request
from jarvis_inference.gemma4 import render_gemma4_prompt
from jarvis_inference.strict_json import parse_body
from jarvis_inference.windows_primary_fixture import (
    FROZEN_FIXTURE_SHA256,
    WindowsPrimaryFixtureFailure,
    derive_fixture_digests,
    load_fixture,
    validate_fixture,
)
from jarvis_inference.windows_primary_report import (
    FROZEN_REPORT_SKELETON_SHA256,
    load_pending_report_skeleton,
    validate_pending_report_skeleton,
)


FIXTURES = Path(__file__).parents[1] / "fixtures"
PERFORMANCE = FIXTURES / "windows-primary-performance-soak-v1.json"
PENDING_REPORT = (
    FIXTURES / "windows-primary-performance-soak-report-pending-v1.json"
)
CAPABILITIES = Capabilities(
    text=True,
    stream=True,
    tools=True,
    json_object=True,
    json_schema=True,
)


def _contains_key(value, key: str) -> bool:
    if isinstance(value, dict):
        return key in value or any(_contains_key(child, key) for child in value.values())
    if isinstance(value, list):
        return any(_contains_key(child, key) for child in value)
    return False


class WindowsPrimaryFixtureTests(unittest.TestCase):
    def test_frozen_fixture_has_exact_balanced_primary_workload(self) -> None:
        fixture = load_fixture(PERFORMANCE)
        cases = fixture.document["performance_cases"]

        self.assertEqual(fixture.fixture_sha256, FROZEN_FIXTURE_SHA256)
        self.assertEqual(len(fixture.requests), 100)
        self.assertEqual(Counter(case["language"] for case in cases), {"el": 50, "en": 50})
        self.assertEqual(
            Counter(case["target_rendered_bin"] for case in cases),
            {512: 34, 2048: 34, 4096: 32},
        )
        self.assertEqual(
            set(Counter(case["variant"] for case in cases).values()),
            {20},
        )
        self.assertEqual(
            fixture.document["execution"]["native_tokenization"],
            {
                "status": "pending_native",
                "exact_counts_claimed": False,
                "tokenizer_identity_source": "qualified_manifest",
                "required_observations": [
                    "generation_serial",
                    "rendered_token_count",
                    "rendered_token_ids_sha256",
                    "accepted_decode_tokens",
                    "accepted_token_ids_sha256",
                    "tokenizer_identity_sha256",
                ],
            },
        )
        self.assertEqual(
            fixture.document["execution"]["native_token_count_ranges"],
            {
                "512": {"minimum": 256, "maximum": 1023},
                "2048": {"minimum": 1024, "maximum": 3071},
                "4096": {"minimum": 3072, "maximum": 6144},
            },
        )
        self.assertEqual(
            fixture.document["execution"]["performance_measurement"],
            {
                "ttft_source": (
                    "gateway_request_start_to_first_accepted_sse_event"
                ),
                "decode_source": "validated_final_llama_timings",
                "end_to_end_source": "client_monotonic_request_to_complete",
                "projected_128_token_formula": (
                    "ttft_seconds + 127 / decode_tokens_per_second"
                ),
            },
        )
        self.assertFalse(_contains_key(fixture.document, "rendered_token_count"))
        self.assertFalse(_contains_key(fixture.document, "rendered_token_ids_sha256"))

    def test_materialized_requests_are_valid_and_renderer_bound(self) -> None:
        fixture = load_fixture(PERFORMANCE)
        rendered_digests = []
        for case, request in zip(
            fixture.document["performance_cases"], fixture.requests, strict=True
        ):
            self.assertEqual(request["max_completion_tokens"], 128)
            self.assertEqual(request["temperature"], 0)
            self.assertEqual(request["top_p"], 1)
            self.assertEqual(request["messages"][0]["role"], "system")
            self.assertGreaterEqual(len(request["messages"]), 4)
            normalized = normalize_request(
                request,
                alias="jarvis-local",
                capabilities=CAPABILITIES,
                context_window=32_768,
            )
            rendered = render_gemma4_prompt(normalized)
            self.assertTrue(rendered.startswith("<bos>"))
            rendered_digests.append(
                hashlib.sha256(rendered.encode("utf-8")).hexdigest()
            )
            if case["variant"] == "json_schema":
                self.assertFalse(request["stream"])
                self.assertEqual(request["response_format"]["type"], "json_schema")
                self.assertNotIn("tools", request)
            elif case["variant"].startswith("tool_"):
                self.assertEqual(len(request["tools"]), 1)
                self.assertIn("tool_choice", request)
            else:
                self.assertTrue(request["stream"])
                self.assertNotIn("tools", request)
        self.assertEqual(len(set(rendered_digests)), 100)
        self.assertEqual(
            derive_fixture_digests(fixture.document),
            (
                fixture.expanded_requests_sha256,
                fixture.rendered_prompt_texts_sha256,
            ),
        )

    def test_soak_and_primary_acceptance_contracts_are_not_softened(self) -> None:
        fixture = load_fixture(PERFORMANCE).document
        soak = fixture["soak"]
        acceptance = fixture["acceptance"]

        self.assertEqual(soak["status"], "pending_native")
        self.assertGreaterEqual(soak["minimum_monotonic_seconds"], 8 * 60 * 60)
        self.assertGreaterEqual(soak["minimum_cycles"], 96)
        self.assertEqual(soak["full_restart_every_cycles"], 12)
        self.assertGreaterEqual(soak["minimum_full_restarts"], 8)
        self.assertEqual(
            soak["cycle_operations"],
            [
                "request_set",
                "per_request_slot_cleanup",
                "combined_gateway_engine_working_set_sample",
            ],
        )
        self.assertEqual(
            soak["memory_sampling"],
            {
                "scope": "combined_gateway_and_engine_working_set",
                "minimum_samples_per_completed_epoch": 12,
                "maximum_epoch_growth_fraction": Decimal("0.05"),
                "maximum_global_growth_fraction": Decimal("0.05"),
                "growth_estimator": "robust_windows_plus_theil_sen",
            },
        )
        self.assertTrue(
            all(
                value == 0
                for value in acceptance["soak"]["maximum_event_counts"].values()
            )
        )
        self.assertEqual(
            acceptance["performance"],
            {
                "timed_requests_required": 100,
                "p95_ttft_seconds_max": 3,
                "projected_128_token_e2e_p95_seconds_max": 10,
                "p5_accepted_decode_tokens_per_second_min": 25,
                "p95_cold_load_to_ready_seconds_max": 120,
            },
        )
        self.assertEqual(
            acceptance["windows_coexistence"]["minimum_free_vram_mib"], 2048
        )
        self.assertEqual(
            acceptance["windows_coexistence"]["maximum_llm_p95_regression_ratio"],
            Decimal("0.15"),
        )
        self.assertEqual(
            acceptance["windows_coexistence"]["minimum_greek_language_correct_per_100"],
            95,
        )
        self.assertEqual(
            acceptance["soak"]["maximum_epoch_working_set_growth_fraction"],
            Decimal("0.05"),
        )
        self.assertEqual(
            acceptance["soak"]["maximum_global_working_set_growth_fraction"],
            Decimal("0.05"),
        )

    def test_tampering_and_pre_native_token_claims_fail_closed(self) -> None:
        raw = PERFORMANCE.read_bytes()
        with tempfile.TemporaryDirectory() as directory:
            tampered = Path(directory) / PERFORMANCE.name
            tampered.write_bytes(raw.replace(b"wp-el-001", b"wp-el-999", 1))
            with self.assertRaisesRegex(
                WindowsPrimaryFixtureFailure, "fixture digest mismatch"
            ):
                load_fixture(tampered)

        document = copy.deepcopy(parse_body(raw))
        document["execution"]["native_tokenization"]["exact_counts_claimed"] = True
        with self.assertRaisesRegex(
            WindowsPrimaryFixtureFailure, "execution contract mismatch"
        ):
            validate_fixture(document)

    def test_symlinked_fixture_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            link = Path(directory) / "fixture.json"
            try:
                link.symlink_to(PERFORMANCE)
            except (NotImplementedError, OSError):
                self.skipTest("symlinks are unavailable")
            with self.assertRaisesRegex(WindowsPrimaryFixtureFailure, "unsafe fixture"):
                load_fixture(link)

    def test_pending_report_is_hash_pinned_sanitized_and_non_evidence(self) -> None:
        fixture = load_fixture(PERFORMANCE)
        report = load_pending_report_skeleton(PENDING_REPORT, fixture)

        self.assertEqual(
            hashlib.sha256(PENDING_REPORT.read_bytes()).hexdigest(),
            FROZEN_REPORT_SKELETON_SHA256,
        )
        self.assertEqual(report["status"], "pending_native")
        self.assertFalse(report["authoritative_qualification"])
        self.assertFalse(report["production_ready"])
        self.assertTrue(all(value is None for value in report["native_identity"].values()))
        self.assertTrue(
            all(value is None for value in report["performance"]["metrics"].values())
        )
        self.assertEqual(report["performance"]["per_case_observations"], [])
        for forbidden in (
            "messages",
            "raw_output",
            "response",
            "tool_arguments",
        ):
            self.assertFalse(_contains_key(report, forbidden))

    def test_pending_report_rejects_fabricated_results(self) -> None:
        fixture = load_fixture(PERFORMANCE)
        report = parse_body(PENDING_REPORT.read_bytes())
        report["status"] = "pass"
        report["production_ready"] = True
        report["performance"]["metrics"]["ttft_p95_seconds"] = 1

        with self.assertRaisesRegex(
            WindowsPrimaryFixtureFailure, "pending report identity mismatch"
        ):
            validate_pending_report_skeleton(report, fixture)

    def test_no_digest_placeholder_remains(self) -> None:
        for value in (
            FROZEN_FIXTURE_SHA256,
            FROZEN_REPORT_SKELETON_SHA256,
        ):
            self.assertEqual(len(value), 64)
            self.assertTrue(all(character in "0123456789abcdef" for character in value))
            self.assertNotEqual(value, "0" * 64)


if __name__ == "__main__":
    unittest.main()
