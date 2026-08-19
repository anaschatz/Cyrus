from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

from _bootstrap import ROOT  # noqa: F401

from jarvis_orchestrator.cloud_contract import (
    DEEPSEEK_MODEL,
    TERRA_MODEL,
    QualifiedEndpointV1,
)
from jarvis_orchestrator.cloud_privacy import JarvisPrivacyClassifierV1
from jarvis_orchestrator.cloud_qualification import (
    CLOUD_RELEASE_VERSION,
    CloudQualificationError,
    dedicated_key_sha256,
    load_cloud_release_qualification,
    run_cloud_privacy_qualification,
)
from jarvis_orchestrator.config import OrchestratorConfig
from jarvis_orchestrator.jcs import domain_sha256


NOW = datetime(2026, 8, 11, 12, 0, tzinfo=UTC)
TOKEN = "sk-or-v1-dedicated-test-key"


def endpoint(model: str) -> QualifiedEndpointV1:
    terra = model == TERRA_MODEL
    return QualifiedEndpointV1(
        model_canonical_id=model,
        provider_endpoint_tag="openai/full" if terra else "deepinfra/full",
        quantization="fp16" if terra else "fp8",
        supported_parameters=("max_completion_tokens", "reasoning")
        if terra
        else ("max_tokens", "reasoning", "temperature", "top_p"),
        qualification_version="m1c-live/1",
        qualification_sha256=("b" if terra else "a") * 64,
        context_limit=60_000 if terra else 128_000,
        output_limit=2_000 if terra else 2_048,
        prompt_rate_per_m="2.50" if terra else "0.09",
        completion_rate_per_m="15" if terra else "0.18",
        request_rate="0",
        runtime_provider_name="OpenAI" if terra else "DeepInfra",
    )


def endpoint_evidence(selected: QualifiedEndpointV1) -> dict[str, object]:
    return {
        "actual_dispatch_count": 1,
        "cost_reconciled": True,
        "model_canonical_id": selected.model_canonical_id,
        "no_model_fallback": True,
        "no_provider_fallback": True,
        "parameter_profile_sha256": selected.parameter_profile_sha256,
        "provider_endpoint_tag": selected.provider_endpoint_tag,
        "qualification_sha256": selected.qualification_sha256,
        "qualification_version": selected.qualification_version,
        "quantization": selected.quantization,
        "receipt_identity_verified": True,
        "request_snapshot_sha256": "c" * 64,
        "response_snapshot_sha256": "d" * 64,
        "zdr_catalog_verified": True,
    }


def bundle(endpoints: tuple[QualifiedEndpointV1, ...]) -> dict[str, object]:
    privacy = JarvisPrivacyClassifierV1()
    return {
        "endpoint_evidence": [endpoint_evidence(item) for item in endpoints],
        "faults": {
            "disconnect_passed": True,
            "missing_usage_passed": True,
            "no_duplicate_dispatch_passed": True,
            "stale_catalog_passed": True,
            "timeout_passed": True,
        },
        "key": {
            "dedicated": True,
            "include_byok_in_limit": False,
            "is_management_key": False,
            "is_provisioning_key": False,
            "key_sha256": dedicated_key_sha256(TOKEN),
            "limit_microusd": 5_000_000,
            "limit_reset": "monthly",
            "observed_at": "2026-08-11T12:00:00Z",
        },
        "privacy": {
            "classifier_version": "jarvis-privacy/1",
            "english_case_count": 10,
            "failure_count": 0,
            "greek_case_count": 10,
            "mixed_case_count": 10,
            "policy_sha256": privacy.policy_sha256,
        },
        "qualified_at": "2026-08-11T12:00:00Z",
        "shadow": {
            "budget_violation_count": 0,
            "ended_at": "2026-08-11T11:00:00Z",
            "external_dispatch_count": 0,
            "fallback_count": 0,
            "maximum_ledger_microusd": 4_000_000,
            "projected_reserved_microusd": 4_500_000,
            "started_at": "2026-07-12T11:00:00Z",
            "turn_count": 1000,
        },
        "valid_until": "2026-09-10T12:00:00Z",
        "version": CLOUD_RELEASE_VERSION,
        "voice_approval": {
            "duplicate_dispatch_count": 0,
            "f10_applied_count": 1,
            "f11_denied_count": 1,
            "real_voice_client": True,
        },
    }


class CloudReleaseQualificationTests(unittest.TestCase):
    def test_frozen_greek_english_and_mixed_privacy_fixtures_pass(self) -> None:
        report = run_cloud_privacy_qualification()
        self.assertTrue(report["ready"])
        self.assertEqual(report["failure_count"], 0)
        self.assertGreaterEqual(report["greek_case_count"], 5)
        self.assertGreaterEqual(report["english_case_count"], 5)
        self.assertGreaterEqual(report["mixed_case_count"], 6)

    def load(
        self,
        document: dict[str, object],
        registry: tuple[QualifiedEndpointV1, ...],
        *,
        terra_enabled: bool = False,
        token: str = TOKEN,
    ):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "release.json"
            path.write_text(json.dumps(document), encoding="utf-8")
            return load_cloud_release_qualification(
                path,
                expected_sha256=domain_sha256(CLOUD_RELEASE_VERSION, document),
                registry=registry,
                bearer_token=token,
                terra_enabled=terra_enabled,
                now=NOW,
            )

    def test_deepseek_is_required_and_terra_is_explicitly_disabled(self) -> None:
        deepseek = endpoint(DEEPSEEK_MODEL)
        terra = endpoint(TERRA_MODEL)
        release = self.load(bundle((deepseek, terra)), (deepseek, terra))
        self.assertEqual(release.active_endpoints, (deepseek,))
        with self.assertRaisesRegex(CloudQualificationError, "deepseek_qualification_required"):
            self.load(bundle((terra,)), (terra,))

    def test_enabled_terra_requires_exact_tuple_evidence(self) -> None:
        deepseek = endpoint(DEEPSEEK_MODEL)
        terra = endpoint(TERRA_MODEL)
        release = self.load(
            bundle((deepseek, terra)),
            (deepseek, terra),
            terra_enabled=True,
        )
        self.assertEqual(release.active_endpoints, (deepseek, terra))
        altered = replace(terra, quantization="bf16")
        with self.assertRaisesRegex(
            CloudQualificationError, "cloud_endpoint_qualification_mismatch"
        ):
            self.load(
                bundle((deepseek, terra)),
                (deepseek, altered),
                terra_enabled=True,
            )

    def test_key_fault_shadow_and_digest_evidence_fail_closed(self) -> None:
        deepseek = endpoint(DEEPSEEK_MODEL)
        document = bundle((deepseek,))
        with self.assertRaisesRegex(CloudQualificationError, "cloud_key_evidence_invalid"):
            self.load(document, (deepseek,), token="different-key")
        document["faults"]["missing_usage_passed"] = False  # type: ignore[index]
        with self.assertRaisesRegex(CloudQualificationError, "cloud_fault_evidence_invalid"):
            self.load(document, (deepseek,))
        document = bundle((deepseek,))
        document["shadow"]["started_at"] = "2026-07-20T11:00:00Z"  # type: ignore[index]
        with self.assertRaisesRegex(CloudQualificationError, "cloud_shadow_evidence_invalid"):
            self.load(document, (deepseek,))

    def test_cloud_config_requires_release_artifact_and_explicit_terra_flag(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = {
                "database_path": str(root / "state.sqlite3"),
                "control_bearer_token": "control",
                "inference_bearer_token": "inference",
                "cloud_enabled": True,
                "cloud_secret_file": str(root / "secret"),
                "cloud_qualification_file": str(root / "registry.json"),
            }
            with self.assertRaisesRegex(ValueError, "cloud_release_qualification_file"):
                OrchestratorConfig.from_test_mapping(base)
            valid = OrchestratorConfig.from_test_mapping(
                {
                    **base,
                    "cloud_release_qualification_file": str(root / "release.json"),
                    "cloud_release_qualification_sha256": "e" * 64,
                    "cloud_terra_enabled": False,
                }
            )
            self.assertFalse(valid.cloud_terra_enabled)


if __name__ == "__main__":
    unittest.main()
