from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import replace
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import _bootstrap  # noqa: F401
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from jarvis_orchestrator.cli import main
from jarvis_orchestrator.config import OrchestratorConfig
from jarvis_orchestrator.operations import OperationalError
from jarvis_orchestrator.readiness import evaluate_operational_ready_gate
from jarvis_orchestrator.readiness_evidence import (
    AGGREGATE_KIND,
    LEAF_EVIDENCE_KINDS,
    LIVE_KIND,
    NATIVE_KIND,
    READINESS_AGGREGATE_VERSION,
    RECOVERY_KIND,
    RESTART_KIND,
    SOAK_KIND,
    SYNTHETIC_KIND,
    AuthenticatedInferenceIdentityV2,
    HostClassV1,
    ReadinessIdentityV2,
    build_readiness_evidence,
    canonical_json_bytes,
    load_ed25519_signer,
    load_native_gate_payload,
    load_pinned_ed25519_verifier,
    qualification_binding_for_kind,
    readiness_config_profile_sha256,
    readiness_identity_from_config,
    sha256_regular_artifact,
    verify_readiness_evidence,
    write_readiness_evidence,
)

NOW = datetime(2026, 8, 18, 12, 0, tzinfo=UTC)
COHORT_ID = "00000000-0000-4000-8000-000000000001"
INFERENCE_MANIFEST_SHA256 = "e" * 64
RUNTIME_IDENTITY_SHA256 = "d" * 64


def _identity(*, build: str = "a" * 64, config: str = "b" * 64) -> ReadinessIdentityV2:
    return ReadinessIdentityV2(
        source_build_sha256=build,
        config_profile_sha256=config,
        model_identity="jarvis-local",
        model_digest_sha256="c" * 64,
        inference_manifest_sha256=INFERENCE_MANIFEST_SHA256,
        inference_runtime_identity_sha256=RUNTIME_IDENTITY_SHA256,
        database_schema_version=9,
        host_class=HostClassV1("macos", "arm64", "source"),
    )


def _healthy_diagnostics(*, ready: bool = True) -> dict[str, object]:
    invariants = {
        "cloud_dispatch_overflow": 0,
        "completed_without_done": 0,
        "duplicate_speech_units": 0,
        "duplicate_text_chunks": 0,
        "multiple_active_conversations": 0,
        "multiple_active_tasks": 0,
        "mutually_exclusive_terminals": 0,
        "output_after_terminal": 0,
    }
    return {
        "active_task_count": 0,
        "claimed_outbox_count": 0,
        "conversation_count": 0,
        "database_bytes": 1,
        "enabled_cloud_endpoint_count": 0,
        "extension_outbox_count": 0,
        "freelist_count": 0,
        "integrity_issue_count": 0 if ready else 1,
        "integrity_ok": ready,
        "invariant_counts": invariants,
        "invariant_violation_count": 0,
        "lifecycle_event_count": 0,
        "operational_log_bytes": 0,
        "outbox_count": 0,
        "page_count": 1,
        "pending_approval_count": 0,
        "pending_capability_reservation_count": 0,
        "pending_model_reservation_count": 0,
        "ready": ready,
        "replayable_extension_outbox_count": 0,
        "replayable_outbox_count": 0,
        "schema_version": 9,
        "shm_bytes": 0,
        "task_count": 0,
        "version": "operational-diagnostics/1",
        "wal_bytes": 0,
    }


def _qualification(kind: str, *, cohort_id: str = COHORT_ID):
    return qualification_binding_for_kind(
        kind,
        cohort_id=cohort_id,
        window_started_at=NOW,
    )


def _config_runtime_identity() -> AuthenticatedInferenceIdentityV2:
    return AuthenticatedInferenceIdentityV2(
        model_alias="jarvis-local",
        manifest_digest=INFERENCE_MANIFEST_SHA256,
        runtime_identity_sha256=RUNTIME_IDENTITY_SHA256,
    )


def _diagnostics_object():
    payload = _healthy_diagnostics()
    return SimpleNamespace(
        schema_version=9,
        to_dict=lambda: dict(payload),
    )


def _synthetic_payload() -> dict[str, object]:
    return {
        "cancellation_stage_counts": {
            "ADMITTED": 129,
            "ROUTING": 129,
            "LOADING_CONTEXT": 129,
            "EXECUTING": 129,
            "RESPONDING": 129,
            "AWAITING_APPROVAL": 129,
            "DECISION_WAITING_RECONNECT": 126,
        },
        "completed_turn_count": 100,
        "elapsed_ms": 10,
        "invariant_check_count": 1_000,
        "invariant_violation_count": 0,
        "maximum_database_bytes": 10_000,
        "ready": True,
        "turn_count": 1_000,
        "version": "synthetic-lifecycle-ready-gate/1",
    }


def _leaf_payloads() -> dict[str, dict[str, object]]:
    from jarvis_voice import PROTOCOL_VERSION, __version__
    from jarvis_voice.native_qualification import SCHEMA_VERSION
    from jarvis_voice.voice_auth import AUTH_PROTOCOL

    native_checks: dict[str, dict[str, object]] = {
        "native_provenance": {
            "code": "qualification.native_hardware_verified",
            "status": "ready",
        },
        "runtime_cases": {
            "code": "qualification.native_cases_passed",
            "status": "ready",
        },
        "turn_integrity": {
            "code": "qualification.native_turn_integrity_passed",
            "duplicate_transcripts": 0,
            "duplicate_tts_playbacks": 0,
            "minimum_real_voice_turns": 200,
            "real_voice_turns": 200,
            "status": "ready",
            "stuck_ptt": 0,
        },
        "acoustic_barge_in": {
            "code": "qualification.native_acoustic_barge_in_passed",
            "maximum_ms": 100.0,
            "samples": 20,
            "status": "ready",
            "threshold_ms": 200.0,
        },
        "automatic_recovery": {
            "code": "qualification.native_recovery_passed",
            "status": "ready",
        },
        "soak": {
            "code": "qualification.native_soak_passed",
            "crashes": 0,
            "duration_seconds": 28_800,
            "memory_stable": True,
            "minimum_duration_seconds": 28_800,
            "rss_samples": 97,
            "status": "ready",
        },
        "runtime_profile": {
            "code": "qualification.native_profile_stable",
            "status": "ready",
        },
        "privacy": {
            "code": "qualification.native_privacy_passed",
            "status": "ready",
        },
    }
    return {
        SYNTHETIC_KIND: _synthetic_payload(),
        LIVE_KIND: {
            "database_bounded": True,
            "duplicate_publication_count": 0,
            "elapsed_ms": 1,
            "maximum_turn_ms": 1,
            "physical_speaker_playback_observed": False,
            "playback_completion_mode": "protocol_probe",
            "ready": True,
            "response_count": 100,
            "speech_unit_count": 100,
            "turn_count": 100,
            "voice_auth_protocol": "jarvis.voice-auth/1",
            "version": "live-sequential-ready-gate/2",
        },
        RESTART_KIND: {
            "duplicate_publication_count": 0,
            "generation_interrupted": True,
            "playback_replay_exact": True,
            "ready": True,
            "voice_auth_protocol": "jarvis.voice-auth/1",
            "version": "live-restart-ready-gate/2",
        },
        RECOVERY_KIND: {
            "backup_integrity_ok": True,
            "corruption_detected": True,
            "elapsed_ms": 1,
            "production_database_mutated": False,
            "ready": True,
            "restore_integrity_ok": True,
            "schema_version": 9,
            "version": "sqlite-recovery-ready-gate/1",
        },
        SOAK_KIND: {
            "batch_count": 1,
            "database_bounded": True,
            "duration_seconds": 86_400,
            "invariant_violation_count": 0,
            "maximum_database_bytes": 1,
            "maximum_operational_log_bytes": 1,
            "operational_log_bounded": True,
            "physical_speaker_playback_observed": False,
            "playback_completion_mode": "protocol_probe",
            "ready": True,
            "turn_count": 100,
            "voice_auth_protocol": "jarvis.voice-auth/1",
            "version": "live-soak-ready-gate/2",
        },
        NATIVE_KIND: {
            "checks": native_checks,
            "code": "qualification.native_ready",
            "content_captured": False,
            "network_used": False,
            "platform": "macos",
            "product_version": __version__,
            "protocol_version": PROTOCOL_VERSION,
            "ready": True,
            "runtime_mode": "source",
            "runtime_profile_sha256": "e" * 64,
            "schema_version": SCHEMA_VERSION,
            "voice_auth_protocol": AUTH_PROTOCOL,
        },
    }


class ReadinessEvidenceTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        private_key = Ed25519PrivateKey.generate()
        self.private_path = self.root / "signing.pem"
        self.private_path.write_bytes(
            private_key.private_bytes(
                serialization.Encoding.PEM,
                serialization.PrivateFormat.PKCS8,
                serialization.NoEncryption(),
            )
        )
        os.chmod(self.private_path, 0o600)
        self.public_path = self.root / "public.pem"
        self.public_path.write_bytes(
            private_key.public_key().public_bytes(
                serialization.Encoding.PEM,
                serialization.PublicFormat.SubjectPublicKeyInfo,
            )
        )
        raw_public = private_key.public_key().public_bytes(
            serialization.Encoding.Raw,
            serialization.PublicFormat.Raw,
        )
        self.public_sha256 = hashlib.sha256(raw_public).hexdigest()
        self.signer = load_ed25519_signer(
            self.private_path,
            key_id="operator-2026",
            expected_public_key_sha256=self.public_sha256,
        )
        self.verifier = load_pinned_ed25519_verifier(
            self.public_path,
            key_id="operator-2026",
            expected_public_key_sha256=self.public_sha256,
        )
        self.build_artifact = self.root / "build.manifest"
        self.build_artifact.write_bytes(b"qualified build")
        self.model_artifact = self.root / "model.manifest"
        self.model_artifact.write_bytes(b"qualified model")
        self.cohort_state = self.root / "cohort-state"

    def config(self, *, control_token: str = "control-fixture") -> OrchestratorConfig:
        return OrchestratorConfig.from_test_mapping(
            {
                "database_path": str(self.root / "state.sqlite3"),
                "control_bearer_token": control_token,
                "inference_bearer_token": "inference-fixture",
                "readiness_signing_key_file": str(self.private_path),
                "readiness_trusted_public_key_file": str(self.public_path),
                "readiness_signer_key_id": "operator-2026",
                "readiness_trusted_public_key_sha256": self.public_sha256,
                "readiness_build_artifact_file": str(self.build_artifact),
                "readiness_build_artifact_sha256": hashlib.sha256(
                    b"qualified build"
                ).hexdigest(),
                "readiness_model_artifact_file": str(self.model_artifact),
                "readiness_model_artifact_sha256": hashlib.sha256(
                    b"qualified model"
                ).hexdigest(),
                "readiness_inference_manifest_sha256": (INFERENCE_MANIFEST_SHA256),
                "readiness_inference_runtime_identity_sha256": (
                    RUNTIME_IDENTITY_SHA256
                ),
            }
        )

    def evidence(self, *, identity: ReadinessIdentityV2 | None = None):
        return build_readiness_evidence(
            evidence_kind=SYNTHETIC_KIND,
            payload=_synthetic_payload(),
            identity=identity or _identity(),
            database_diagnostics=_healthy_diagnostics(),
            qualification=_qualification(SYNTHETIC_KIND),
            signer=self.signer,
            captured_at=NOW,
            freshness_seconds=3_600,
        )

    def leaf_paths(
        self,
        *,
        cohort_by_kind: dict[str, str] | None = None,
        captured_by_kind: dict[str, datetime] | None = None,
        payload_by_kind: dict[str, dict[str, object]] | None = None,
        set_name: str = "leaf-set",
        cohort_state_directory: Path | None = None,
    ) -> dict[str, Path]:
        selected_cohorts = cohort_by_kind or {}
        selected_captures = captured_by_kind or {}
        selected_payloads = _leaf_payloads()
        selected_payloads.update(payload_by_kind or {})
        output_directory = self.root / set_name
        result: dict[str, Path] = {}
        for kind, payload in selected_payloads.items():
            evidence = build_readiness_evidence(
                evidence_kind=kind,
                payload=payload,
                identity=_identity(),
                database_diagnostics=_healthy_diagnostics(),
                qualification=_qualification(
                    kind,
                    cohort_id=selected_cohorts.get(kind, COHORT_ID),
                ),
                signer=self.signer,
                captured_at=selected_captures.get(kind, NOW),
                freshness_seconds=7 * 24 * 60 * 60,
            )
            result[kind] = write_readiness_evidence(
                output_directory / f"{kind}.json",
                evidence,
                verifier=self.verifier,
                cohort_state_directory=(cohort_state_directory or self.cohort_state),
            )
        return result

    def test_signed_canonical_round_trip_requires_external_verifier(self) -> None:
        path = write_readiness_evidence(
            self.root / "evidence.json",
            self.evidence(),
            verifier=self.verifier,
            cohort_state_directory=self.cohort_state,
        )

        verified = verify_readiness_evidence(
            path,
            verifier=self.verifier,
            expected_kind=SYNTHETIC_KIND,
            expected_identity=_identity(),
            now=NOW + timedelta(minutes=1),
        )

        self.assertEqual(verified.payload["ready"], True)
        self.assertEqual(path.read_bytes(), canonical_json_bytes(verified.canonical()))
        self.assertNotIn("PRIVATE", path.read_text(encoding="utf-8"))
        if os.name == "posix":
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)

    def test_tamper_stale_and_untrusted_self_signature_fail_closed(self) -> None:
        path = write_readiness_evidence(
            self.root / "evidence.json",
            self.evidence(),
            verifier=self.verifier,
            cohort_state_directory=self.cohort_state,
        )
        tampered = json.loads(path.read_text(encoding="utf-8"))
        tampered["payload"]["turn_count"] = 100_000
        path.write_bytes(canonical_json_bytes(tampered))
        with self.assertRaisesRegex(
            OperationalError,
            "readiness_evidence_(?:invalid|signature_invalid)",
        ):
            verify_readiness_evidence(
                path,
                verifier=self.verifier,
                expected_kind=SYNTHETIC_KIND,
                expected_identity=_identity(),
                now=NOW,
            )

        path = write_readiness_evidence(
            self.root / "evidence.json",
            self.evidence(),
            verifier=self.verifier,
            cohort_state_directory=self.cohort_state,
        )
        with self.assertRaisesRegex(OperationalError, "readiness_evidence_stale"):
            verify_readiness_evidence(
                path,
                verifier=self.verifier,
                expected_kind=SYNTHETIC_KIND,
                expected_identity=_identity(),
                now=NOW + timedelta(hours=1),
            )

        attacker = Ed25519PrivateKey.generate()
        from jarvis_orchestrator.readiness_evidence import Ed25519EvidenceSigner

        self_signed = build_readiness_evidence(
            evidence_kind=SYNTHETIC_KIND,
            payload=_synthetic_payload(),
            identity=_identity(),
            database_diagnostics=_healthy_diagnostics(),
            qualification=_qualification(SYNTHETIC_KIND),
            signer=Ed25519EvidenceSigner("operator-2026", attacker),
            captured_at=NOW,
        )
        with self.assertRaisesRegex(
            OperationalError, "readiness_evidence_signature_invalid"
        ):
            write_readiness_evidence(
                self.root / "self-signed.json",
                self_signed,
                verifier=self.verifier,
                cohort_state_directory=self.cohort_state,
            )
        path = self.root / "self-signed.json"
        path.write_bytes(canonical_json_bytes(self_signed.canonical()))
        with self.assertRaisesRegex(
            OperationalError, "readiness_evidence_signature_invalid"
        ):
            verify_readiness_evidence(
                path,
                verifier=self.verifier,
                expected_kind=SYNTHETIC_KIND,
                expected_identity=_identity(),
                now=NOW,
            )

    def test_cross_build_and_cross_config_are_rejected(self) -> None:
        path = write_readiness_evidence(
            self.root / "evidence.json",
            self.evidence(),
            verifier=self.verifier,
            cohort_state_directory=self.cohort_state,
        )
        for expected in (
            _identity(build="d" * 64),
            _identity(config="e" * 64),
        ):
            with (
                self.subTest(expected=expected),
                self.assertRaisesRegex(
                    OperationalError, "readiness_evidence_identity_mismatch"
                ),
            ):
                verify_readiness_evidence(
                    path,
                    verifier=self.verifier,
                    expected_kind=SYNTHETIC_KIND,
                    expected_identity=expected,
                    now=NOW,
                )

    def test_duplicate_extra_and_legacy_documents_are_ineligible(self) -> None:
        path = self.root / "bad.json"
        path.write_text('{"version":"x","version":"x"}', encoding="utf-8")
        with self.assertRaisesRegex(OperationalError, "readiness_evidence_invalid"):
            verify_readiness_evidence(
                path,
                verifier=self.verifier,
                expected_kind=SYNTHETIC_KIND,
                expected_identity=_identity(),
                now=NOW,
            )

        extra = self.evidence().canonical()
        extra["caller_ready_override"] = True
        path.write_bytes(canonical_json_bytes(extra))
        with self.assertRaisesRegex(OperationalError, "readiness_evidence_invalid"):
            verify_readiness_evidence(
                path,
                verifier=self.verifier,
                expected_kind=SYNTHETIC_KIND,
                expected_identity=_identity(),
                now=NOW,
            )

        path.write_bytes(canonical_json_bytes(_synthetic_payload()))
        with self.assertRaisesRegex(
            OperationalError, "readiness_evidence_legacy_ineligible"
        ):
            verify_readiness_evidence(
                path,
                verifier=self.verifier,
                expected_kind=SYNTHETIC_KIND,
                expected_identity=_identity(),
                now=NOW,
            )

    def test_malformed_envelope_enums_fail_before_signature_verification(self) -> None:
        valid = self.evidence().canonical()
        path = self.root / "malformed-enum.json"
        verifier = SimpleNamespace(verify=Mock(side_effect=AssertionError("unexpected verification")))
        for field in ("evidence_kind", "os_family", "architecture", "runtime_mode"):
            for invalid in ([], {}):
                with self.subTest(field=field, invalid=invalid):
                    document = deepcopy(valid)
                    target = document if field == "evidence_kind" else document["host_class"]
                    target[field] = invalid
                    path.write_bytes(canonical_json_bytes(document))
                    with self.assertRaises(OperationalError):
                        verify_readiness_evidence(
                            path, verifier=verifier, expected_kind=SYNTHETIC_KIND,
                            expected_identity=_identity(), now=NOW,
                        )
                    verifier.verify.assert_not_called()

    def test_malformed_native_enums_fail_before_signing(self) -> None:
        signer = SimpleNamespace(key_id="fixture", sign=Mock(side_effect=AssertionError("unexpected signing")))
        for field in ("platform", "runtime_mode", "code", "check_status", "check_code"):
            for invalid in ([], {}):
                with self.subTest(field=field, invalid=invalid):
                    payload = _leaf_payloads()[NATIVE_KIND]
                    if field.startswith("check_"):
                        payload["checks"]["acoustic_barge_in"][field.removeprefix("check_")] = invalid
                    else:
                        payload[field] = invalid
                    with self.assertRaisesRegex(OperationalError, "readiness_payload_invalid"):
                        build_readiness_evidence(
                            evidence_kind=NATIVE_KIND, payload=payload, identity=_identity(),
                            database_diagnostics=_healthy_diagnostics(),
                            qualification=_qualification(NATIVE_KIND), signer=signer, captured_at=NOW,
                        )
                    signer.sign.assert_not_called()

    def test_qualification_rejects_nonstring_evidence_kind(self) -> None:
        binding = _qualification(SYNTHETIC_KIND)
        for invalid in ([], {}):
            for action in (
                lambda: qualification_binding_for_kind(invalid, cohort_id=COHORT_ID, window_started_at=NOW),
                lambda: binding.validate_for(invalid, binding.window_started_at_utc),
            ):
                with self.subTest(invalid=invalid, action=action), self.assertRaises(OperationalError):
                    action()

    def test_clock_inputs_and_utc_overflow_fail_with_operational_errors(self) -> None:
        path = self.root / "clock-evidence.json"
        path.write_bytes(canonical_json_bytes(self.evidence().canonical()))
        arguments = dict(
            evidence_kind=SYNTHETIC_KIND, payload=_synthetic_payload(), identity=_identity(),
            database_diagnostics=_healthy_diagnostics(), qualification=_qualification(SYNTHETIC_KIND),
            signer=self.signer,
        )
        aggregate = build_readiness_evidence(
            **{**arguments, "evidence_kind": AGGREGATE_KIND,
               "qualification": _qualification(AGGREGATE_KIND),
               "payload": {"checks": {kind: {"ready": True} for kind in LEAF_EVIDENCE_KINDS},
                           "content_captured": False, "ready": True, "version": READINESS_AGGREGATE_VERSION}},
            captured_at=NOW,
            parent_evidence_sha256s={kind: hashlib.sha256(kind.encode()).hexdigest() for kind in LEAF_EVIDENCE_KINDS},
        )
        invalid_times = ([], True, "not-a-clock", NOW.replace(tzinfo=None),
                         datetime.min.replace(tzinfo=timezone(timedelta(hours=14))))
        for value in invalid_times:
            for operation, action in (
                ("build", lambda: build_readiness_evidence(**arguments, captured_at=value)),
                ("verify", lambda: verify_readiness_evidence(path, verifier=self.verifier,
                    expected_kind=SYNTHETIC_KIND, expected_identity=_identity(), now=value)),
                ("publish", lambda: write_readiness_evidence(self.root / "invalid-time-aggregate.json", aggregate,
                    verifier=self.verifier, cohort_state_directory=self.cohort_state,
                    current_state_provider=lambda: (_identity(), _healthy_diagnostics()), final_time_provider=lambda: value)),
            ):
                with self.subTest(operation=operation, value=value), self.assertRaises(OperationalError):
                    action()
        self.assertFalse((self.root / "invalid-time-aggregate.json").exists())
        self.assertFalse(self.cohort_state.exists())
        for started in (datetime.max.replace(tzinfo=UTC), invalid_times[-1]):
            with self.subTest(started=started), self.assertRaises(OperationalError):
                qualification_binding_for_kind(SYNTHETIC_KIND, cohort_id=COHORT_ID, window_started_at=started)
        with self.subTest(case="expiry_overflow"), self.assertRaises(OperationalError):
            build_readiness_evidence(
                **{**arguments, "qualification": qualification_binding_for_kind(
                    SYNTHETIC_KIND, cohort_id=COHORT_ID, window_started_at="9999-12-27T00:00:00Z")},
                captured_at=datetime(9999, 12, 29, tzinfo=UTC), freshness_seconds=7 * 24 * 60 * 60,
            )
        with self.subTest(case="maximum_verification_time"), self.assertRaisesRegex(OperationalError, "readiness_evidence_stale"):
            verify_readiness_evidence(path, verifier=self.verifier, expected_kind=SYNTHETIC_KIND,
                                      expected_identity=_identity(), now=datetime.max.replace(tzinfo=UTC))

    def test_clock_normalization_preserves_offsets_and_exact_expiry(self) -> None:
        evidence = self.evidence()
        path = self.root / "offset-evidence.json"
        path.write_bytes(canonical_json_bytes(evidence.canonical()))
        offset = timezone(timedelta(hours=5, minutes=30))
        for now in (NOW.astimezone(offset), NOW + timedelta(hours=1, microseconds=-1)):
            self.assertEqual(verify_readiness_evidence(
                path, verifier=self.verifier, expected_kind=SYNTHETIC_KIND,
                expected_identity=_identity(), now=now,
            ), evidence)
        self.assertEqual(qualification_binding_for_kind(
            SYNTHETIC_KIND, cohort_id=COHORT_ID, window_started_at=NOW.astimezone(offset),
        ), _qualification(SYNTHETIC_KIND))
        with self.assertRaisesRegex(OperationalError, "readiness_evidence_stale"):
            verify_readiness_evidence(path, verifier=self.verifier, expected_kind=SYNTHETIC_KIND,
                                      expected_identity=_identity(), now=NOW + timedelta(hours=1))

    def test_aggregate_requires_exact_parent_hash_set(self) -> None:
        parents = {
            kind: hashlib.sha256(kind.encode()).hexdigest()
            for kind in LEAF_EVIDENCE_KINDS
        }
        checks = {kind: {"ready": True} for kind in LEAF_EVIDENCE_KINDS}
        aggregate = build_readiness_evidence(
            evidence_kind=AGGREGATE_KIND,
            payload={
                "checks": checks,
                "content_captured": False,
                "ready": True,
                "version": READINESS_AGGREGATE_VERSION,
            },
            identity=_identity(),
            database_diagnostics=_healthy_diagnostics(),
            qualification=_qualification(AGGREGATE_KIND),
            signer=self.signer,
            captured_at=NOW,
            freshness_seconds=3_600,
            parent_evidence_sha256s=parents,
        )
        path = self.root / "aggregate.json"
        path.write_bytes(canonical_json_bytes(aggregate.canonical()))
        os.chmod(path, 0o600)
        verify_readiness_evidence(
            path,
            verifier=self.verifier,
            expected_kind=AGGREGATE_KIND,
            expected_identity=_identity(),
            expected_parent_evidence_sha256s=parents,
            now=NOW,
        )
        verify_readiness_evidence(
            path,
            verifier=self.verifier,
            expected_kind=AGGREGATE_KIND,
            expected_identity=_identity(),
            now=NOW,
        )
        missing = dict(parents)
        missing.pop(LEAF_EVIDENCE_KINDS[-1])
        with self.assertRaisesRegex(
            OperationalError, "readiness_parent_evidence_invalid"
        ):
            verify_readiness_evidence(
                path,
                verifier=self.verifier,
                expected_kind=AGGREGATE_KIND,
                expected_identity=_identity(),
                expected_parent_evidence_sha256s=missing,
                now=NOW,
            )
        with self.assertRaisesRegex(
            OperationalError, "readiness_parent_evidence_invalid"
        ):
            build_readiness_evidence(
                evidence_kind=AGGREGATE_KIND,
                payload=aggregate.payload,
                identity=_identity(),
                database_diagnostics=_healthy_diagnostics(),
                qualification=_qualification(AGGREGATE_KIND),
                signer=self.signer,
                captured_at=NOW,
                parent_evidence_sha256s=missing,
            )

    @unittest.skipUnless(os.name == "posix", "POSIX mode assertion")
    def test_signing_key_must_be_owner_private(self) -> None:
        key = Ed25519PrivateKey.generate()
        path = self.root / "exposed.pem"
        path.write_bytes(
            key.private_bytes(
                serialization.Encoding.PEM,
                serialization.PrivateFormat.PKCS8,
                serialization.NoEncryption(),
            )
        )
        os.chmod(path, 0o644)
        with self.assertRaisesRegex(
            OperationalError, "readiness_signing_key_not_private"
        ):
            load_ed25519_signer(
                path,
                key_id="operator-2026",
                expected_public_key_sha256=self.public_sha256,
            )

    def test_artifact_digest_rejects_mutation_and_symlink(self) -> None:
        artifact = self.root / "artifact.bin"
        artifact.write_bytes(b"build manifest")
        self.assertEqual(
            sha256_regular_artifact(artifact),
            hashlib.sha256(b"build manifest").hexdigest(),
        )
        real_read = os.read
        mutated = False

        def mutate_after_read(descriptor: int, maximum: int) -> bytes:
            nonlocal mutated
            content = real_read(descriptor, maximum)
            if content and not mutated:
                mutated = True
                artifact.write_bytes(b"other manifest")
            return content

        with (
            patch(
                "jarvis_orchestrator.readiness_evidence.os.read",
                side_effect=mutate_after_read,
            ),
            self.assertRaisesRegex(OperationalError, "readiness_artifact_invalid"),
        ):
            sha256_regular_artifact(artifact)
        link = self.root / "artifact-link.bin"
        try:
            link.symlink_to(artifact)
        except OSError:
            self.skipTest("symlinks unavailable")
        with self.assertRaisesRegex(OperationalError, "readiness_artifact_invalid"):
            sha256_regular_artifact(link)

    def test_native_projection_is_strict_content_free_data(self) -> None:
        from jarvis_voice.native_qualification import native_evidence_template

        raw = native_evidence_template(
            "macos",
            runtime_profile_sha256="a" * 64,
        )
        raw_path = self.root / "native.json"
        raw_path.write_text(json.dumps(raw, indent=2), encoding="utf-8")
        projected = load_native_gate_payload(
            raw_path,
            expected_runtime_profile_sha256="a" * 64,
            expected_host_class=HostClassV1("macos", "arm64", "source"),
        )
        self.assertFalse(projected["ready"])
        self.assertEqual(projected["runtime_mode"], "source")
        self.assertEqual(projected["runtime_profile_sha256"], "a" * 64)
        self.assertNotIn(raw["run_id"], canonical_json_bytes(projected).decode())

        duplicate = self.root / "duplicate-native.json"
        duplicate.write_text(
            '{"schema_version":2,"schema_version":2}',
            encoding="utf-8",
        )
        with self.assertRaisesRegex(OperationalError, "readiness_evidence_invalid"):
            load_native_gate_payload(
                duplicate,
                expected_runtime_profile_sha256="a" * 64,
                expected_host_class=HostClassV1("macos", "arm64", "source"),
            )

    def test_write_revalidates_mutated_nested_payload(self) -> None:
        evidence = self.evidence()
        counts = evidence.payload["cancellation_stage_counts"]
        assert isinstance(counts, dict)
        counts["ADMITTED"] = "caller-authored secret"
        with self.assertRaisesRegex(OperationalError, "readiness_payload_invalid"):
            write_readiness_evidence(
                self.root / "malformed.json",
                evidence,
                verifier=self.verifier,
                cohort_state_directory=self.cohort_state,
            )
        self.assertFalse((self.root / "malformed.json").exists())
        with self.assertRaisesRegex(OperationalError, "readiness_evidence_invalid"):
            replace(self.evidence(), signature="not-a-signature")

        native = _leaf_payloads()[NATIVE_KIND]
        native_checks = native["checks"]
        assert isinstance(native_checks, dict)
        acoustic = native_checks["acoustic_barge_in"]
        assert isinstance(acoustic, dict)
        acoustic["maximum_ms"] = 10**400
        with self.assertRaisesRegex(OperationalError, "readiness_payload_invalid"):
            build_readiness_evidence(
                evidence_kind=NATIVE_KIND,
                payload=native,
                identity=_identity(),
                database_diagnostics=_healthy_diagnostics(),
                qualification=_qualification(NATIVE_KIND),
                signer=self.signer,
                captured_at=NOW,
            )

    def test_config_profile_is_redacted_and_artifacts_are_cross_checked(self) -> None:
        first = self.config(control_token="DO_NOT_EMIT_FIRST")
        second = self.config(control_token="DO_NOT_EMIT_SECOND")
        self.assertNotEqual(
            readiness_config_profile_sha256(first),
            readiness_config_profile_sha256(second),
        )
        identity = readiness_identity_from_config(
            first,
            database_schema_version=9,
            inference_identity=_config_runtime_identity(),
        )
        evidence = build_readiness_evidence(
            evidence_kind=SYNTHETIC_KIND,
            payload=_synthetic_payload(),
            identity=identity,
            database_diagnostics=_healthy_diagnostics(),
            qualification=_qualification(SYNTHETIC_KIND),
            signer=self.signer,
            captured_at=NOW,
        )
        encoded = canonical_json_bytes(evidence.canonical()).decode("utf-8")
        self.assertNotIn("DO_NOT_EMIT_FIRST", encoded)
        self.assertNotIn("inference-fixture", encoded)
        self.model_artifact.write_bytes(b"mutated model")
        with self.assertRaisesRegex(
            OperationalError, "readiness_model_identity_mismatch"
        ):
            readiness_identity_from_config(
                first,
                database_schema_version=9,
                inference_identity=_config_runtime_identity(),
            )

    def test_enabled_memory_requires_and_binds_live_crypto_state(self) -> None:
        config = replace(self.config(), memory_enabled=True)
        with self.assertRaisesRegex(
            OperationalError, "readiness_memory_crypto_required"
        ):
            readiness_identity_from_config(
                config,
                database_schema_version=10,
                inference_identity=_config_runtime_identity(),
            )
        first = readiness_identity_from_config(
            config,
            database_schema_version=10,
            inference_identity=_config_runtime_identity(),
            memory_crypto_state_sha256="1" * 64,
        )
        second = readiness_identity_from_config(
            config,
            database_schema_version=10,
            inference_identity=_config_runtime_identity(),
            memory_crypto_state_sha256="2" * 64,
        )
        self.assertNotEqual(first, second)
        evidence = build_readiness_evidence(
            evidence_kind=SYNTHETIC_KIND,
            payload=_synthetic_payload(),
            identity=first,
            database_diagnostics={**_healthy_diagnostics(), "schema_version": 10},
            qualification=_qualification(SYNTHETIC_KIND),
            signer=self.signer,
            captured_at=NOW,
        )
        path = write_readiness_evidence(
            self.root / "memory-bound.json",
            evidence,
            verifier=self.verifier,
            cohort_state_directory=self.cohort_state,
        )
        with self.assertRaisesRegex(
            OperationalError, "readiness_evidence_identity_mismatch"
        ):
            verify_readiness_evidence(
                path,
                verifier=self.verifier,
                expected_kind=SYNTHETIC_KIND,
                expected_identity=second,
                now=NOW,
            )

    def test_config_rejects_partial_or_unpinned_readiness_authority(self) -> None:
        base = {
            "database_path": str(self.root / "state.sqlite3"),
            "control_bearer_token": "control-fixture",
            "inference_bearer_token": "inference-fixture",
            "readiness_signing_key_file": str(self.private_path),
        }
        with self.assertRaisesRegex(
            ValueError, "readiness evidence configuration is incomplete"
        ):
            OrchestratorConfig.from_test_mapping(base)
        complete = self.config()
        with self.assertRaisesRegex(ValueError, "readiness evidence digest is invalid"):
            replace(
                complete,
                readiness_trusted_public_key_sha256="A" * 64,
            ).validate()
        production = {
            "database_path": str(self.root / "production.sqlite3"),
            "control_bearer_token": base64.urlsafe_b64encode(b"c" * 32)
            .rstrip(b"=")
            .decode("ascii"),
            "inference_bearer_token": base64.urlsafe_b64encode(b"i" * 32)
            .rstrip(b"=")
            .decode("ascii"),
            "readiness_signing_key_file": str(self.private_path),
            "readiness_trusted_public_key_file": str(self.public_path),
            "readiness_signer_key_id": "operator-2026",
            "readiness_trusted_public_key_sha256": self.public_sha256,
            "readiness_build_artifact_file": str(self.build_artifact),
            "readiness_build_artifact_sha256": hashlib.sha256(
                b"qualified build"
            ).hexdigest(),
            "readiness_model_artifact_file": str(self.model_artifact),
            "readiness_model_artifact_sha256": hashlib.sha256(
                b"qualified model"
            ).hexdigest(),
            "readiness_inference_manifest_sha256": INFERENCE_MANIFEST_SHA256,
            "readiness_inference_runtime_identity_sha256": (RUNTIME_IDENTITY_SHA256),
        }
        self.assertEqual(
            OrchestratorConfig.from_mapping(production).readiness_signer_key_id,
            "operator-2026",
        )

    def test_operational_aggregate_binds_exact_leaf_kind_to_hash(self) -> None:
        paths: dict[str, Path] = {}
        leaves = {}
        for kind, payload in _leaf_payloads().items():
            evidence = build_readiness_evidence(
                evidence_kind=kind,
                payload=payload,
                identity=_identity(),
                database_diagnostics=_healthy_diagnostics(),
                qualification=_qualification(kind),
                signer=self.signer,
                captured_at=NOW,
                freshness_seconds=3_600,
            )
            leaves[kind] = evidence
            paths[kind] = write_readiness_evidence(
                self.root / f"{kind}.json",
                evidence,
                verifier=self.verifier,
                cohort_state_directory=self.cohort_state,
            )

        aggregate = evaluate_operational_ready_gate(
            synthetic_evidence=paths[SYNTHETIC_KIND],
            live_evidence=paths[LIVE_KIND],
            restart_evidence=paths[RESTART_KIND],
            recovery_evidence=paths[RECOVERY_KIND],
            soak_evidence=paths[SOAK_KIND],
            native_voice_evidence=paths[NATIVE_KIND],
            verifier=self.verifier,
            expected_identity=_identity(),
            aggregate_signer=self.signer,
            cohort_state_directory=self.cohort_state,
            current_state_provider=lambda: (
                _identity(),
                _healthy_diagnostics(),
            ),
            final_time_provider=lambda: NOW + timedelta(seconds=1),
            now=NOW,
        )
        expected_parents = {
            kind: leaves[kind].evidence_sha256 for kind in LEAF_EVIDENCE_KINDS
        }
        self.assertTrue(aggregate.payload["ready"])
        self.assertEqual(dict(aggregate.parent_evidence_sha256s), expected_parents)
        aggregate_path = write_readiness_evidence(
            self.root / "aggregate.json",
            aggregate,
            verifier=self.verifier,
            cohort_state_directory=self.cohort_state,
            current_state_provider=lambda: (
                _identity(),
                _healthy_diagnostics(),
            ),
            final_time_provider=lambda: NOW + timedelta(seconds=1),
        )
        verify_readiness_evidence(
            aggregate_path,
            verifier=self.verifier,
            expected_kind=AGGREGATE_KIND,
            expected_identity=_identity(),
            expected_parent_evidence_sha256s=expected_parents,
            now=NOW,
        )
        with self.assertRaisesRegex(
            OperationalError, "readiness_evidence_identity_mismatch"
        ):
            evaluate_operational_ready_gate(
                synthetic_evidence=paths[LIVE_KIND],
                live_evidence=paths[SYNTHETIC_KIND],
                restart_evidence=paths[RESTART_KIND],
                recovery_evidence=paths[RECOVERY_KIND],
                soak_evidence=paths[SOAK_KIND],
                native_voice_evidence=paths[NATIVE_KIND],
                verifier=self.verifier,
                expected_identity=_identity(),
                aggregate_signer=self.signer,
                cohort_state_directory=self.cohort_state,
                current_state_provider=lambda: (
                    _identity(),
                    _healthy_diagnostics(),
                ),
                final_time_provider=lambda: NOW + timedelta(seconds=1),
                now=NOW,
            )

        self_declared_payload = _synthetic_payload()
        self_declared_payload["cancellation_stage_counts"] = {
            stage: 1 for stage in self_declared_payload["cancellation_stage_counts"]
        }
        self_declared_state = self.root / "self-declared-cohort-state"
        refused_paths = self.leaf_paths(
            payload_by_kind={SYNTHETIC_KIND: self_declared_payload},
            set_name="self-declared",
            cohort_state_directory=self_declared_state,
        )
        refused = evaluate_operational_ready_gate(
            synthetic_evidence=refused_paths[SYNTHETIC_KIND],
            live_evidence=refused_paths[LIVE_KIND],
            restart_evidence=refused_paths[RESTART_KIND],
            recovery_evidence=refused_paths[RECOVERY_KIND],
            soak_evidence=refused_paths[SOAK_KIND],
            native_voice_evidence=refused_paths[NATIVE_KIND],
            verifier=self.verifier,
            expected_identity=_identity(),
            aggregate_signer=self.signer,
            cohort_state_directory=self_declared_state,
            current_state_provider=lambda: (
                _identity(),
                _healthy_diagnostics(),
            ),
            final_time_provider=lambda: NOW + timedelta(seconds=1),
            now=NOW,
        )
        self.assertFalse(refused.payload["ready"])
        self.assertFalse(refused.payload["checks"][SYNTHETIC_KIND]["ready"])

    def test_aggregate_rechecks_and_binds_current_database_health(self) -> None:
        paths = self.leaf_paths()
        common = {
            "synthetic_evidence": paths[SYNTHETIC_KIND],
            "live_evidence": paths[LIVE_KIND],
            "restart_evidence": paths[RESTART_KIND],
            "recovery_evidence": paths[RECOVERY_KIND],
            "soak_evidence": paths[SOAK_KIND],
            "native_voice_evidence": paths[NATIVE_KIND],
            "verifier": self.verifier,
            "expected_identity": _identity(),
            "aggregate_signer": self.signer,
            "cohort_state_directory": self.cohort_state,
            "final_time_provider": lambda: NOW + timedelta(seconds=1),
            "now": NOW,
        }
        with self.assertRaisesRegex(
            OperationalError,
            "readiness_database_unhealthy",
        ):
            evaluate_operational_ready_gate(
                **common,
                current_state_provider=lambda: (
                    _identity(),
                    _healthy_diagnostics(ready=False),
                ),
            )

        aggregate = evaluate_operational_ready_gate(
            **common,
            current_state_provider=lambda: (
                _identity(),
                _healthy_diagnostics(),
            ),
        )
        self.assertTrue(aggregate.database_diagnostics["ready"])
        self.assertTrue(aggregate.database_diagnostics["integrity_ok"])
        self.assertEqual(
            aggregate.database_diagnostics["invariant_violation_count"],
            0,
        )
        self.assertEqual(len(aggregate.database_diagnostics_sha256), 64)
        changed = _healthy_diagnostics()
        changed["database_bytes"] = 2
        output = self.root / "changed-database-aggregate.json"
        with self.assertRaisesRegex(
            OperationalError,
            "readiness_database_state_changed",
        ):
            write_readiness_evidence(
                output,
                aggregate,
                verifier=self.verifier,
                cohort_state_directory=self.cohort_state,
                current_state_provider=lambda: (_identity(), changed),
            )
        self.assertFalse(output.exists())

    def test_runtime_alias_manifest_and_runtime_pin_must_match(self) -> None:
        config = self.config()
        failures = (
            AuthenticatedInferenceIdentityV2(
                "other-local",
                _config_runtime_identity().manifest_digest,
                RUNTIME_IDENTITY_SHA256,
            ),
            AuthenticatedInferenceIdentityV2(
                "jarvis-local",
                "f" * 64,
                RUNTIME_IDENTITY_SHA256,
            ),
            AuthenticatedInferenceIdentityV2(
                "jarvis-local",
                _config_runtime_identity().manifest_digest,
                "f" * 64,
            ),
        )
        for runtime_identity in failures:
            with (
                self.subTest(runtime_identity=runtime_identity),
                self.assertRaisesRegex(
                    OperationalError,
                    "readiness_runtime_identity_mismatch",
                ),
            ):
                readiness_identity_from_config(
                    config,
                    database_schema_version=9,
                    inference_identity=runtime_identity,
                )

        paths = self.leaf_paths()
        with self.assertRaisesRegex(
            OperationalError,
            "readiness_identity_changed_during_gate",
        ):
            evaluate_operational_ready_gate(
                synthetic_evidence=paths[SYNTHETIC_KIND],
                live_evidence=paths[LIVE_KIND],
                restart_evidence=paths[RESTART_KIND],
                recovery_evidence=paths[RECOVERY_KIND],
                soak_evidence=paths[SOAK_KIND],
                native_voice_evidence=paths[NATIVE_KIND],
                verifier=self.verifier,
                expected_identity=_identity(),
                aggregate_signer=self.signer,
                cohort_state_directory=self.cohort_state,
                current_state_provider=lambda: (
                    replace(
                        _identity(),
                        inference_runtime_identity_sha256="f" * 64,
                    ),
                    _healthy_diagnostics(),
                ),
                final_time_provider=lambda: NOW + timedelta(seconds=1),
                now=NOW,
            )

    def test_native_projection_rejects_cross_platform_mode_and_profile(self) -> None:
        from jarvis_voice.native_qualification import native_evidence_template

        raw = native_evidence_template(
            "macos",
            runtime_profile_sha256="a" * 64,
        )
        path = self.root / "cross-native.json"
        path.write_text(json.dumps(raw), encoding="utf-8")
        mismatches = (
            ("a" * 64, HostClassV1("windows", "x86_64", "signed_package")),
            ("a" * 64, HostClassV1("macos", "arm64", "signed_package")),
            ("b" * 64, HostClassV1("macos", "arm64", "source")),
        )
        for profile_sha256, host_class in mismatches:
            with (
                self.subTest(host_class=host_class),
                self.assertRaisesRegex(
                    OperationalError,
                    "native_voice_evidence_identity_mismatch",
                ),
            ):
                load_native_gate_payload(
                    path,
                    expected_runtime_profile_sha256=profile_sha256,
                    expected_host_class=host_class,
                )

    def test_aggregate_rejects_mixed_and_replayed_cohorts(self) -> None:
        different_cohort = "00000000-0000-4000-8000-000000000002"
        mixed = self.leaf_paths(
            cohort_by_kind={NATIVE_KIND: different_cohort},
        )
        with self.assertRaisesRegex(
            OperationalError,
            "readiness_qualification_cohort_invalid",
        ):
            evaluate_operational_ready_gate(
                synthetic_evidence=mixed[SYNTHETIC_KIND],
                live_evidence=mixed[LIVE_KIND],
                restart_evidence=mixed[RESTART_KIND],
                recovery_evidence=mixed[RECOVERY_KIND],
                soak_evidence=mixed[SOAK_KIND],
                native_voice_evidence=mixed[NATIVE_KIND],
                verifier=self.verifier,
                expected_identity=_identity(),
                aggregate_signer=self.signer,
                cohort_state_directory=self.cohort_state,
                current_state_provider=lambda: (
                    _identity(),
                    _healthy_diagnostics(),
                ),
                final_time_provider=lambda: NOW + timedelta(seconds=1),
                now=NOW,
            )

        replayed_state = self.root / "replayed-cohort-state"
        replayed = self.leaf_paths(
            captured_by_kind={SYNTHETIC_KIND: NOW + timedelta(minutes=1)},
            set_name="replayed",
            cohort_state_directory=replayed_state,
        )
        with self.assertRaisesRegex(
            OperationalError,
            "readiness_qualification_cohort_invalid",
        ):
            evaluate_operational_ready_gate(
                synthetic_evidence=replayed[SYNTHETIC_KIND],
                live_evidence=replayed[LIVE_KIND],
                restart_evidence=replayed[RESTART_KIND],
                recovery_evidence=replayed[RECOVERY_KIND],
                soak_evidence=replayed[SOAK_KIND],
                native_voice_evidence=replayed[NATIVE_KIND],
                verifier=self.verifier,
                expected_identity=_identity(),
                aggregate_signer=self.signer,
                cohort_state_directory=replayed_state,
                current_state_provider=lambda: (
                    _identity(),
                    _healthy_diagnostics(),
                ),
                final_time_provider=lambda: NOW + timedelta(minutes=1, seconds=1),
                now=NOW + timedelta(minutes=1),
            )

        with self.assertRaisesRegex(
            OperationalError,
            "readiness_qualification_invalid",
        ):
            build_readiness_evidence(
                evidence_kind=LIVE_KIND,
                payload=_leaf_payloads()[LIVE_KIND],
                identity=_identity(),
                database_diagnostics=_healthy_diagnostics(),
                qualification=replace(_qualification(LIVE_KIND), sequence=1),
                signer=self.signer,
                captured_at=NOW,
            )

    def test_aggregate_rejects_future_leaf_at_final_signing_time(self) -> None:
        state = self.root / "future-leaf-state"
        paths = self.leaf_paths(
            captured_by_kind={NATIVE_KIND: NOW + timedelta(seconds=20)},
            set_name="future-leaf",
            cohort_state_directory=state,
        )
        with self.assertRaisesRegex(
            OperationalError,
            "readiness_qualification_window_invalid",
        ):
            evaluate_operational_ready_gate(
                synthetic_evidence=paths[SYNTHETIC_KIND],
                live_evidence=paths[LIVE_KIND],
                restart_evidence=paths[RESTART_KIND],
                recovery_evidence=paths[RECOVERY_KIND],
                soak_evidence=paths[SOAK_KIND],
                native_voice_evidence=paths[NATIVE_KIND],
                verifier=self.verifier,
                expected_identity=_identity(),
                aggregate_signer=self.signer,
                cohort_state_directory=state,
                current_state_provider=lambda: (
                    _identity(),
                    _healthy_diagnostics(),
                ),
                final_time_provider=lambda: NOW,
                now=NOW,
            )

    def test_aggregate_rejects_provider_delay_past_cohort_window(self) -> None:
        state = self.root / "delayed-provider-state"
        paths = self.leaf_paths(
            set_name="delayed-provider",
            cohort_state_directory=state,
        )
        calls: list[str] = []

        def current_state():
            calls.append("state")
            return _identity(), _healthy_diagnostics()

        def final_time() -> datetime:
            calls.append("clock")
            return NOW + timedelta(hours=72)

        with self.assertRaisesRegex(
            OperationalError,
            "readiness_qualification_window_invalid",
        ):
            evaluate_operational_ready_gate(
                synthetic_evidence=paths[SYNTHETIC_KIND],
                live_evidence=paths[LIVE_KIND],
                restart_evidence=paths[RESTART_KIND],
                recovery_evidence=paths[RECOVERY_KIND],
                soak_evidence=paths[SOAK_KIND],
                native_voice_evidence=paths[NATIVE_KIND],
                verifier=self.verifier,
                expected_identity=_identity(),
                aggregate_signer=self.signer,
                cohort_state_directory=state,
                current_state_provider=current_state,
                final_time_provider=final_time,
                now=NOW,
            )
        self.assertEqual(calls, ["state", "clock"])

    def test_identical_leaf_set_produces_one_idempotent_aggregate(self) -> None:
        state = self.root / "idempotent-aggregate-state"
        paths = self.leaf_paths(
            set_name="idempotent-aggregate",
            cohort_state_directory=state,
        )
        arguments = {
            "synthetic_evidence": paths[SYNTHETIC_KIND],
            "live_evidence": paths[LIVE_KIND],
            "restart_evidence": paths[RESTART_KIND],
            "recovery_evidence": paths[RECOVERY_KIND],
            "soak_evidence": paths[SOAK_KIND],
            "native_voice_evidence": paths[NATIVE_KIND],
            "verifier": self.verifier,
            "expected_identity": _identity(),
            "aggregate_signer": self.signer,
            "cohort_state_directory": state,
            "current_state_provider": lambda: (
                _identity(),
                _healthy_diagnostics(),
            ),
            "final_time_provider": lambda: NOW + timedelta(seconds=1),
            "now": NOW,
        }
        first = evaluate_operational_ready_gate(**arguments)
        second = evaluate_operational_ready_gate(**arguments)
        self.assertEqual(first.captured_at_utc, "2026-08-18T12:00:01Z")
        self.assertEqual(
            canonical_json_bytes(first.canonical()),
            canonical_json_bytes(second.canonical()),
        )
        output = self.root / "idempotent-aggregate.json"
        first_path = write_readiness_evidence(
            output,
            first,
            verifier=self.verifier,
            cohort_state_directory=state,
            current_state_provider=lambda: (
                _identity(),
                _healthy_diagnostics(),
            ),
            final_time_provider=lambda: NOW + timedelta(seconds=1),
        )
        first_bytes = first_path.read_bytes()
        second_path = write_readiness_evidence(
            output,
            second,
            verifier=self.verifier,
            cohort_state_directory=state,
            current_state_provider=lambda: (
                _identity(),
                _healthy_diagnostics(),
            ),
            final_time_provider=lambda: NOW + timedelta(seconds=1),
        )
        self.assertEqual(second_path.read_bytes(), first_bytes)
        with self.assertRaisesRegex(
            OperationalError,
            "readiness_qualification_sequence_claimed",
        ):
            write_readiness_evidence(
                self.root / "distinct-aggregate-replay.json",
                second,
                verifier=self.verifier,
                cohort_state_directory=state,
                current_state_provider=lambda: (
                    _identity(),
                    _healthy_diagnostics(),
                ),
                final_time_provider=lambda: NOW + timedelta(seconds=1),
            )

    def test_duplicate_leaf_sequence_conflicts_and_exact_retry_recovers(self) -> None:
        state = self.root / "duplicate-leaf-state"
        evidence = self.evidence()
        output = self.root / "claimed-leaf.json"
        with (
            patch(
                "jarvis_orchestrator.readiness_evidence._atomic_write_evidence",
                side_effect=OperationalError("simulated_write_interruption"),
            ),
            self.assertRaisesRegex(
                OperationalError,
                "simulated_write_interruption",
            ),
        ):
            write_readiness_evidence(
                output,
                evidence,
                verifier=self.verifier,
                cohort_state_directory=state,
            )
        self.assertFalse(output.exists())
        changed_payload = _synthetic_payload()
        changed_payload["elapsed_ms"] = 11
        changed_evidence = build_readiness_evidence(
            evidence_kind=SYNTHETIC_KIND,
            payload=changed_payload,
            identity=_identity(),
            database_diagnostics=_healthy_diagnostics(),
            qualification=_qualification(SYNTHETIC_KIND),
            signer=self.signer,
            captured_at=NOW,
            freshness_seconds=3_600,
        )
        with self.assertRaisesRegex(
            OperationalError,
            "readiness_qualification_sequence_claimed",
        ):
            write_readiness_evidence(
                output,
                changed_evidence,
                verifier=self.verifier,
                cohort_state_directory=state,
            )
        recovered = write_readiness_evidence(
            output,
            evidence,
            verifier=self.verifier,
            cohort_state_directory=state,
        )
        self.assertEqual(
            recovered.read_bytes(),
            canonical_json_bytes(evidence.canonical()),
        )
        with self.assertRaisesRegex(
            OperationalError,
            "readiness_qualification_sequence_claimed",
        ):
            write_readiness_evidence(
                self.root / "duplicate-leaf.json",
                evidence,
                verifier=self.verifier,
                cohort_state_directory=state,
            )
        receipt = next(state.glob("readiness-cohort-*.json"))
        if os.name == "posix":
            self.assertEqual(receipt.stat().st_mode & 0o777, 0o600)

    def test_destination_alias_is_canonical_and_cannot_claim_two_slots(self) -> None:
        physical = self.root / "physical-evidence"
        physical.mkdir(mode=0o700)
        alias = self.root / "evidence-alias"
        try:
            alias.symlink_to(physical, target_is_directory=True)
        except OSError:
            self.skipTest("directory symlinks unavailable")
        state = self.root / "alias-state"
        synthetic = self.evidence()
        first = write_readiness_evidence(
            physical / "shared.json",
            synthetic,
            verifier=self.verifier,
            cohort_state_directory=state,
        )
        retry = write_readiness_evidence(
            alias / "shared.json",
            synthetic,
            verifier=self.verifier,
            cohort_state_directory=state,
        )
        self.assertEqual(first, retry)
        live = build_readiness_evidence(
            evidence_kind=LIVE_KIND,
            payload=_leaf_payloads()[LIVE_KIND],
            identity=_identity(),
            database_diagnostics=_healthy_diagnostics(),
            qualification=_qualification(LIVE_KIND),
            signer=self.signer,
            captured_at=NOW,
        )
        with self.assertRaisesRegex(
            OperationalError,
            "readiness_qualification_destination_claimed",
        ):
            write_readiness_evidence(
                alias / "shared.json",
                live,
                verifier=self.verifier,
                cohort_state_directory=state,
            )

    def test_invalid_private_receipt_has_one_content_free_error(self) -> None:
        state = self.root / "invalid-receipt-state"
        evidence = self.evidence()
        output = self.root / "invalid-receipt-leaf.json"
        write_readiness_evidence(
            output,
            evidence,
            verifier=self.verifier,
            cohort_state_directory=state,
        )
        receipt = next(state.glob("readiness-cohort-*.json"))
        if os.name == "posix":
            os.chmod(receipt, 0o644)
        else:  # pragma: no cover - native Windows exercises the DACL branch
            receipt.write_bytes(b"{}")
        with self.assertRaises(OperationalError) as caught:
            write_readiness_evidence(
                output,
                evidence,
                verifier=self.verifier,
                cohort_state_directory=state,
            )
        self.assertEqual(caught.exception.code, "readiness_cohort_receipt_invalid")
        self.assertEqual(str(caught.exception), "readiness_cohort_receipt_invalid")

    def test_cli_requires_configured_trust_root_and_writes_signed_envelope(
        self,
    ) -> None:
        unsigned_config = OrchestratorConfig.from_test_mapping(
            {
                "database_path": str(self.root / "unsigned.sqlite3"),
                "control_bearer_token": "control-fixture",
                "inference_bearer_token": "inference-fixture",
            }
        )
        refused_output = self.root / "refused.json"
        qualification_window_started = datetime.now(UTC).replace(
            microsecond=0
        ) - timedelta(minutes=1)
        qualification_args = [
            "--qualification-cohort-id",
            COHORT_ID,
            "--qualification-window-started-at-utc",
            qualification_window_started.isoformat().replace("+00:00", "Z"),
        ]
        stdout = io.StringIO()
        with (
            patch(
                "jarvis_orchestrator.cli.OrchestratorConfig.load",
                return_value=unsigned_config,
            ),
            patch(
                "jarvis_orchestrator.cli.run_synthetic_lifecycle_gate",
                return_value=_synthetic_payload(),
            ) as run_gate,
            redirect_stdout(stdout),
        ):
            status = main(
                [
                    "--config",
                    "/unused/config.json",
                    "operations",
                    "synthetic-gate",
                    *qualification_args,
                    "--output",
                    str(refused_output),
                ]
            )
        self.assertEqual(status, 1)
        self.assertEqual(
            json.loads(stdout.getvalue())["error"]["code"],
            "readiness_authority_required",
        )
        self.assertFalse(refused_output.exists())
        run_gate.assert_not_called()

        config = self.config(control_token="DO_NOT_PRINT_CLI")
        mismatched = replace(
            config,
            readiness_trusted_public_key_sha256="d" * 64,
        )
        stdout = io.StringIO()
        with (
            patch(
                "jarvis_orchestrator.cli.OrchestratorConfig.load",
                return_value=mismatched,
            ),
            patch(
                "jarvis_orchestrator.cli.run_synthetic_lifecycle_gate",
                return_value=_synthetic_payload(),
            ) as run_gate,
            patch(
                "jarvis_orchestrator.cli.database_diagnostics",
                return_value=SimpleNamespace(schema_version=9),
            ),
            redirect_stdout(stdout),
        ):
            status = main(
                [
                    "--config",
                    "/unused/config.json",
                    "operations",
                    "synthetic-gate",
                    *qualification_args,
                    "--output",
                    str(self.root / "mismatched.json"),
                ]
            )
        self.assertEqual(status, 1)
        self.assertEqual(
            json.loads(stdout.getvalue())["error"]["code"],
            "readiness_verifier_pin_mismatch",
        )
        self.assertNotIn("DO_NOT_PRINT_CLI", stdout.getvalue())
        run_gate.assert_not_called()

        output = self.root / "signed.json"
        stdout = io.StringIO()
        with (
            patch(
                "jarvis_orchestrator.cli.OrchestratorConfig.load",
                return_value=config,
            ),
            patch(
                "jarvis_orchestrator.cli.run_synthetic_lifecycle_gate",
                return_value=_synthetic_payload(),
            ),
            patch(
                "jarvis_orchestrator.cli.database_diagnostics",
                return_value=_diagnostics_object(),
            ),
            patch("jarvis_orchestrator.cli.InferenceControlClient") as client,
            redirect_stdout(stdout),
        ):
            client.return_value.runtime_identity.return_value = SimpleNamespace(
                manifest_digest=INFERENCE_MANIFEST_SHA256,
                runtime_identity_sha256=RUNTIME_IDENTITY_SHA256,
            )
            status = main(
                [
                    "--config",
                    "/unused/config.json",
                    "operations",
                    "synthetic-gate",
                    *qualification_args,
                    "--output",
                    str(output),
                ]
            )
        self.assertEqual(status, 0)
        self.assertNotIn("DO_NOT_PRINT_CLI", stdout.getvalue())
        verify_readiness_evidence(
            output,
            verifier=self.verifier,
            expected_kind=SYNTHETIC_KIND,
            expected_identity=readiness_identity_from_config(
                config,
                database_schema_version=9,
                inference_identity=_config_runtime_identity(),
            ),
        )


if __name__ == "__main__":
    unittest.main()
