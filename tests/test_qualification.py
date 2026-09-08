from __future__ import annotations

import json
import hashlib
import subprocess
import tempfile
import unittest
import uuid
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import _bootstrap  # noqa: F401

from jarvis_voice import PROTOCOL_VERSION, __version__
from jarvis_voice.cli import main
from jarvis_voice.config import voice_config_from_mapping
from jarvis_voice.distribution import verify_distribution, write_distribution_manifest
from jarvis_voice.errors import BackendError
from jarvis_voice.native_qualification import native_evidence_template
from jarvis_voice.voice_auth import AUTH_PROTOCOL
from jarvis_voice.qualification import (
    CASE_REQUIREMENTS,
    DISPATCH_LATENCY,
    PLAYBACK_LATENCY,
    PURGE_LATENCY,
    TRANSCRIPT_LATENCY,
    build_release_report,
    evaluate_frozen_doctor,
    evaluate_samples,
    host_qualification,
    load_samples,
    parse_metrics,
    record_sample,
    sample_template,
    validate_samples,
    verified_distribution_manifest_sha256,
    verify_authenticode,
    write_sample_template,
    write_report,
)


def sample_id(index: int) -> str:
    return str(uuid.UUID(int=index + 1, version=4))


def enough_samples() -> list[dict[str, object]]:
    samples: list[dict[str, object]] = []
    index = 0
    for case, minimum in CASE_REQUIREMENTS.items():
        for offset in range(minimum):
            metrics: dict[str, float] = {}
            if case in {"greek_turn", "english_turn"}:
                metrics = {TRANSCRIPT_LATENCY: 100.0 + offset, DISPATCH_LATENCY: 10.0 + offset, PLAYBACK_LATENCY: 5.0 + offset}
            elif case in {"barge_in_first_unit", "barge_in_last_unit"}:
                metrics = {PURGE_LATENCY: 8.0 + offset}
            samples.append({"sample_id": sample_id(index), "case": case, "outcome": "pass", "metrics": metrics})
            index += 1
    return samples


def sample_evidence(
    samples: list[dict[str, object]] | None = None,
    *,
    runtime_profile_sha256: str = "a" * 64,
) -> dict[str, object]:
    evidence = sample_template(runtime_profile_sha256)
    evidence["samples"] = [] if samples is None else samples
    return evidence


def frozen_doctor(*, ready: bool = True) -> dict[str, object]:
    status = "ready" if ready else "blocked"
    return {
        "ready": ready,
        "network_used": False,
        "product_version": __version__,
        "protocol_version": PROTOCOL_VERSION,
        "voice_auth_protocol": AUTH_PROTOCOL,
        "checks": {
            "host": {"status": status},
            "tray": {"status": status},
            "transport": {
                "status": status,
                "code": (
                    "transport.loopback_authenticated"
                    if ready
                    else "transport.authentication_key_required"
                ),
            },
            "packaging": {"status": status, "frozen": True, "bundle_mode": "onedir", "dll_search": {"status": status}, "mutable_data": {"status": status}},
        },
    }


def ready_native_windows(*, runtime_profile_sha256: str = "b" * 64) -> dict[str, object]:
    evidence = native_evidence_template("windows")
    evidence["provenance"] = {
        "native_machine": True,
        "voice_layer_mocks_used": False,
        "synthetic_voice_turns": False,
        "hardware_capture": True,
        "physical_ptt": True,
        "physical_playback": True,
    }
    for value in evidence["cases"].values():
        value.update({"status": "pass", "attempts": 1})
    evidence["counters"].update(
        {
            "real_voice_turns": 200,
            "microphone_disconnect_recoveries": 1,
            "sleep_resume_recoveries": 1,
        }
    )
    evidence["acoustic_barge_in"] = {
        "method": "acoustic_microphone",
        "audio_path": "wired",
        "samples": 20,
        "max_ptt_to_silence_ms": 150.0,
        "silence_hold_ms": 50,
    }
    evidence["soak"] = {
        "duration_seconds": 28_800,
        "rss_samples": 97,
        "rss_start_bytes": 300_000_000,
        "rss_end_bytes": 301_000_000,
        "rss_peak_bytes": 350_000_000,
        "rss_slope_bytes_per_hour": 125_000.0,
    }
    evidence["runtime_profile"] = {
        "mode": "signed_package",
        "fingerprint_sha256": runtime_profile_sha256,
        "stable": True,
        "signed": True,
    }
    return evidence


class QualificationSampleTests(unittest.TestCase):
    def test_evidence_write_failures_remain_typed_when_cleanup_also_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "report.json"
            with patch.object(Path, "replace", side_effect=PermissionError("private replace failure")), patch.object(Path, "unlink", side_effect=PermissionError("private cleanup failure")):
                with self.assertRaisesRegex(BackendError, "qualification.output_write_failed"):
                    write_report(path, {"ready": False})

    def test_unserializable_evidence_never_creates_an_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "report.json"
            for value in ({"metric": float("nan")}, {"value": object()}, {"value": "\ud800"}):
                with self.assertRaisesRegex(BackendError, "qualification.output_write_failed"):
                    write_report(path, value)
                self.assertFalse(path.exists())

    def test_malformed_sample_values_fail_with_a_typed_schema_error(self) -> None:
        changes = (("case", []), ("outcome", {}), ("metrics", {TRANSCRIPT_LATENCY: 10**5000}))
        for field, value in changes:
            sample = {"sample_id": sample_id(0), "case": "no_speech", "outcome": "pass", "metrics": {}}
            sample[field] = value
            with self.subTest(field=field), self.assertRaisesRegex(BackendError, "qualification.samples_schema_invalid"):
                validate_samples(sample_evidence([sample]))
        evidence = sample_evidence()
        evidence["schema_version"] = 2.0
        with self.assertRaisesRegex(BackendError, "qualification.samples_schema_invalid"):
            validate_samples(evidence)

    def test_malformed_sample_json_is_typed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "samples.json"
            for raw in (b"[" * 2000 + b"0" + b"]" * 2000, b"9" * 5000, b"\xff"):
                with self.subTest(prefix=raw[:10]):
                    path.write_bytes(raw)
                    with self.assertRaisesRegex(BackendError, "qualification.samples_unreadable"):
                        load_samples(path)

    def test_evidence_writer_rejects_symlink_before_resolving_it(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "existing.json"
            target.write_bytes(b"original")
            link = Path(temporary) / "alias.json"
            try:
                link.symlink_to(target)
            except (OSError, NotImplementedError):
                self.skipTest("requires symlink support")
            with self.assertRaisesRegex(BackendError, "qualification.output_symlink_forbidden"):
                write_sample_template(link, "a" * 64)
            self.assertEqual(target.read_bytes(), b"original")

    def test_template_is_empty_and_strict(self) -> None:
        expected = {
            "schema_version": 2,
            "product_version": __version__,
            "protocol_version": PROTOCOL_VERSION,
            "voice_auth_protocol": AUTH_PROTOCOL,
            "runtime_profile_sha256": "a" * 64,
            "samples": [],
        }
        self.assertEqual(sample_template("a" * 64), expected)
        with self.assertRaisesRegex(BackendError, "qualification.samples_schema_invalid"):
            validate_samples({**expected, "notes": "secret transcript"})

    def test_legacy_and_wrong_identity_samples_are_rejected(self) -> None:
        with self.assertRaisesRegex(BackendError, "qualification.samples_schema_invalid"):
            validate_samples({"schema_version": 1, "samples": []})
        for field, value in (
            ("product_version", "0.7.0"),
            ("protocol_version", "jarvis.voice/0"),
            ("voice_auth_protocol", "jarvis.voice-auth/0"),
        ):
            evidence = sample_evidence()
            evidence[field] = value
            with self.subTest(field=field), self.assertRaisesRegex(
                BackendError, "qualification.samples_schema_invalid"
            ):
                validate_samples(evidence)

    def test_sample_runtime_profile_binding_is_exact(self) -> None:
        validate_samples(
            sample_evidence(),
            expected_runtime_profile_sha256="a" * 64,
        )
        with self.assertRaisesRegex(
            BackendError, "qualification.samples_runtime_profile_mismatch"
        ):
            validate_samples(
                sample_evidence(),
                expected_runtime_profile_sha256="b" * 64,
            )

    def test_sufficient_samples_produce_only_counts_and_percentiles(self) -> None:
        samples = validate_samples(sample_evidence(enough_samples()))
        report = evaluate_samples(samples)
        self.assertEqual(report["status"], "ready")
        self.assertEqual(report["samples_total"], sum(CASE_REQUIREMENTS.values()))
        self.assertEqual(report["metrics"][TRANSCRIPT_LATENCY]["count"], 20)
        self.assertIn("p50_ms", report["metrics"][PURGE_LATENCY])
        self.assertNotIn("sample_id", json.dumps(report))

    def test_failure_and_insufficient_counts_block_release(self) -> None:
        samples = enough_samples(); samples[0]["outcome"] = "fail"; samples[0]["metrics"] = {}
        self.assertEqual(evaluate_samples(validate_samples(sample_evidence(samples)))["code"], "qualification.failed_sample_present")
        self.assertEqual(evaluate_samples([])["code"], "qualification.samples_insufficient")

    def test_replay_unknown_metric_and_missing_required_metric_are_rejected(self) -> None:
        samples = enough_samples(); samples[1]["sample_id"] = samples[0]["sample_id"]
        with self.assertRaisesRegex(BackendError, "qualification.sample_id_invalid"):
            validate_samples(sample_evidence(samples))
        broken = enough_samples(); broken[0]["metrics"] = {"transcript": 1}
        with self.assertRaisesRegex(BackendError, "qualification.samples_schema_invalid"):
            validate_samples(sample_evidence(broken))
        broken = enough_samples(); broken[0]["metrics"] = {TRANSCRIPT_LATENCY: 1, DISPATCH_LATENCY: 1}
        with self.assertRaisesRegex(BackendError, "qualification.required_metric_missing"):
            validate_samples(sample_evidence(broken))

    def test_non_finite_negative_and_duplicate_cli_metrics_are_rejected(self) -> None:
        for raw in ({TRANSCRIPT_LATENCY: float("nan")}, {TRANSCRIPT_LATENCY: -1}):
            sample = {"sample_id": sample_id(0), "case": "no_speech", "outcome": "pass", "metrics": raw}
            with self.assertRaisesRegex(BackendError, "qualification.samples_schema_invalid"):
                validate_samples(sample_evidence([sample]))
        with self.assertRaisesRegex(BackendError, "qualification.metric_argument_invalid"):
            parse_metrics([f"{TRANSCRIPT_LATENCY}=1", f"{TRANSCRIPT_LATENCY}=2"])

    def test_atomic_record_contains_no_free_form_field(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "samples.json"; write_sample_template(path, "a" * 64)
            fixed = uuid.UUID(int=99, version=4)
            self.assertEqual(record_sample(path, case="no_speech", outcome="pass", metrics={}, id_factory=lambda: fixed), str(fixed))
            self.assertEqual(set(json.loads(path.read_text())["samples"][0]), {"sample_id", "case", "outcome", "metrics"})


class AuthenticodeTests(unittest.TestCase):
    def _artifacts(self, root: Path) -> None:
        (root / "_internal").mkdir()
        (root / "JARVIS Voice.exe").write_bytes(b"exe")
        (root / "_internal" / "runtime.dll").write_bytes(b"dll")
        (root / "_internal" / "module.pyd").write_bytes(b"pyd")
        (root / "README.txt").write_text("not a PE")

    def test_non_windows_and_missing_tool_are_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            self._artifacts(Path(temporary))
            self.assertEqual(verify_authenticode(temporary, platform_name="darwin")["code"], "qualification.authenticode_native_windows_required")
            self.assertEqual(verify_authenticode(temporary, platform_name="win32", which=lambda _: None)["code"], "qualification.signtool_missing")

    def test_every_pe_is_verified_without_shell_and_output_is_discarded(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            self._artifacts(Path(temporary)); calls: list[tuple[list[str], dict[str, object]]] = []
            def runner(command, **kwargs):
                calls.append((command, kwargs)); return SimpleNamespace(returncode=0)
            report = verify_authenticode(temporary, platform_name="win32", which=lambda _: "signtool.exe", runner=runner)
            self.assertEqual(report["status"], "ready"); self.assertEqual(report["files_checked"], 3)
            self.assertTrue(all(call[1]["shell"] is False and call[1]["capture_output"] is True and call[1]["text"] is False for call in calls))
            self.assertTrue(all(call[0][1:5] == ["verify", "/pa", "/all", "/tw"] for call in calls))
            self.assertNotIn(str(Path(temporary)), json.dumps(report))

    def test_warning_and_timeout_are_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            self._artifacts(Path(temporary))
            warning = verify_authenticode(temporary, platform_name="win32", which=lambda _: "signtool", runner=lambda *a, **k: SimpleNamespace(returncode=2))
            self.assertEqual(warning["code"], "qualification.authenticode_invalid")
            timeout = verify_authenticode(temporary, platform_name="win32", which=lambda _: "signtool", runner=lambda *a, **k: (_ for _ in ()).throw(subprocess.TimeoutExpired("signtool", 1)))
            self.assertEqual(timeout["code"], "qualification.signtool_timeout")


class ReleaseReportTests(unittest.TestCase):
    def test_runtime_profile_hash_binds_the_verified_manifest_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "Voice.exe").write_bytes(b"exe")
            manifest = write_distribution_manifest(root)
            expected = hashlib.sha256(manifest.read_bytes()).hexdigest()

            def replace_after_verification(directory):
                result = verify_distribution(directory)
                manifest.write_bytes(b"unverified replacement")
                return result

            with patch("jarvis_voice.qualification.verify_distribution", side_effect=replace_after_verification):
                self.assertEqual(verified_distribution_manifest_sha256(root), expected)

    def test_host_gate_requires_exact_native_profile(self) -> None:
        self.assertEqual(host_qualification(platform_name="win32", build=22_000, pointer_bits=64, python_version=(3, 11))["status"], "ready")
        self.assertEqual(host_qualification(platform_name="win32", build=22_000, pointer_bits=64, python_version=(3, 12))["status"], "blocked")
        self.assertEqual(host_qualification(platform_name="darwin", pointer_bits=64, python_version=(3, 11))["status"], "blocked")

    def test_frozen_doctor_requires_matching_version_and_frozen_gates(self) -> None:
        self.assertEqual(evaluate_frozen_doctor(frozen_doctor())["status"], "ready")
        wrong = frozen_doctor(); wrong["product_version"] = "0.0.0"
        self.assertEqual(evaluate_frozen_doctor(wrong)["status"], "blocked")
        wrong = frozen_doctor(); wrong["checks"]["packaging"]["dll_search"]["status"] = "blocked"
        self.assertEqual(evaluate_frozen_doctor(wrong)["status"], "blocked")

    def test_legacy_unauthenticated_frozen_doctor_cannot_qualify(self) -> None:
        legacy = frozen_doctor()
        legacy.pop("voice_auth_protocol")
        legacy["checks"].pop("transport")
        self.assertEqual(evaluate_frozen_doctor(legacy)["status"], "blocked")

        wrong_protocol = frozen_doctor()
        wrong_protocol["voice_auth_protocol"] = "jarvis.voice-auth/0"
        self.assertEqual(evaluate_frozen_doctor(wrong_protocol)["status"], "blocked")

    def test_all_gates_create_ready_report_without_paths_or_content(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); distribution = root / "dist"; distribution.mkdir()
            (distribution / "JARVIS Voice.exe").write_bytes(b"exe"); write_distribution_manifest(distribution)
            runtime_profile = verified_distribution_manifest_sha256(distribution)
            samples = root / "samples.json"; samples.write_text(json.dumps(sample_evidence(enough_samples(), runtime_profile_sha256=runtime_profile)))
            frozen = root / "frozen.json"; frozen.write_text(json.dumps(frozen_doctor()))
            native = root / "native.json"; native.write_text(json.dumps(ready_native_windows(runtime_profile_sha256=runtime_profile)))
            report = build_release_report(
                voice_config_from_mapping({}), distribution=distribution, samples_path=samples, frozen_doctor_path=frozen, native_evidence_path=native,
                now=lambda: datetime(2026, 8, 10, tzinfo=UTC), host_probe=lambda: {"status": "ready", "code": "host.ready"},
                doctor_runner=lambda _: {
                    "ready": True,
                    "network_used": False,
                    "voice_auth_protocol": AUTH_PROTOCOL,
                    "checks": {
                        "transport": {
                            "status": "ready",
                            "code": "transport.loopback_authenticated",
                        }
                    },
                },
                signature_verifier=lambda _: {"status": "ready", "code": "signature.ready", "files_checked": 1},
            )
            self.assertTrue(report["ready"])
            self.assertEqual(report["schema_version"], 2)
            self.assertEqual(report["voice_auth_protocol"], AUTH_PROTOCOL)
            encoded = json.dumps(report)
            self.assertNotIn(str(root), encoded); self.assertNotIn("μυστικό περιεχόμενο", encoded)
            self.assertFalse(report["network_used"]); self.assertFalse(report["content_captured"])

    def test_release_rejects_samples_and_native_evidence_from_other_distribution(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); distribution = root / "dist"; distribution.mkdir()
            (distribution / "JARVIS Voice.exe").write_bytes(b"exe"); write_distribution_manifest(distribution)
            runtime_profile = verified_distribution_manifest_sha256(distribution)
            frozen = root / "frozen.json"; frozen.write_text(json.dumps(frozen_doctor()))
            samples = root / "samples.json"
            native = root / "native.json"

            def report() -> dict[str, object]:
                return build_release_report(
                    voice_config_from_mapping({}),
                    distribution=distribution,
                    samples_path=samples,
                    frozen_doctor_path=frozen,
                    native_evidence_path=native,
                    host_probe=lambda: {"status": "ready", "code": "host.ready"},
                    doctor_runner=lambda _: {
                        "ready": True,
                        "network_used": False,
                        "voice_auth_protocol": AUTH_PROTOCOL,
                        "checks": {
                            "transport": {
                                "status": "ready",
                                "code": "transport.loopback_authenticated",
                            }
                        },
                    },
                    signature_verifier=lambda _: {
                        "status": "ready",
                        "code": "signature.ready",
                        "files_checked": 1,
                    },
                )

            samples.write_text(json.dumps(sample_evidence(enough_samples(), runtime_profile_sha256="f" * 64)))
            native.write_text(json.dumps(ready_native_windows(runtime_profile_sha256=runtime_profile)))
            samples_mismatch = report()
            self.assertFalse(samples_mismatch["ready"])
            self.assertEqual(
                samples_mismatch["checks"]["samples"]["code"],
                "qualification.samples_runtime_profile_mismatch",
            )

            samples.write_text(json.dumps(sample_evidence(enough_samples(), runtime_profile_sha256=runtime_profile)))
            native.write_text(json.dumps(ready_native_windows(runtime_profile_sha256="f" * 64)))
            native_mismatch = report()
            self.assertFalse(native_mismatch["ready"])
            self.assertEqual(
                native_mismatch["checks"]["native_acceptance"]["code"],
                "qualification.native_profile_mismatch",
            )

    def test_legacy_unauthenticated_source_doctor_blocks_release(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); distribution = root / "dist"; distribution.mkdir()
            (distribution / "JARVIS Voice.exe").write_bytes(b"exe"); write_distribution_manifest(distribution)
            runtime_profile = verified_distribution_manifest_sha256(distribution)
            samples = root / "samples.json"; samples.write_text(json.dumps(sample_evidence(enough_samples(), runtime_profile_sha256=runtime_profile)))
            frozen = root / "frozen.json"; frozen.write_text(json.dumps(frozen_doctor()))
            native = root / "native.json"; native.write_text(json.dumps(ready_native_windows(runtime_profile_sha256=runtime_profile)))
            report = build_release_report(
                voice_config_from_mapping({}), distribution=distribution, samples_path=samples, frozen_doctor_path=frozen, native_evidence_path=native,
                host_probe=lambda: {"status": "ready", "code": "host.ready"},
                doctor_runner=lambda _: {"ready": True, "network_used": False},
                signature_verifier=lambda _: {"status": "ready", "code": "signature.ready", "files_checked": 1},
            )
            self.assertFalse(report["ready"])
            self.assertEqual(report["checks"]["source_doctor"]["status"], "blocked")
            self.assertEqual(
                report["checks"]["source_doctor"]["code"],
                "qualification.source_doctor_not_ready",
            )

    def test_cli_template_record_and_development_host_blocked_report(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            distribution = root / "dist"; distribution.mkdir(); (distribution / "JARVIS Voice.exe").write_bytes(b"unsigned"); write_distribution_manifest(distribution)
            samples = root / "samples.json"
            self.assertEqual(main(["qualification-template", "--distribution", str(distribution), "--output", str(samples)]), 0)
            self.assertEqual(main(["qualification-record", "--samples", str(samples), "--case", "no_speech", "--outcome", "pass"]), 0)
            frozen = root / "frozen.json"; frozen.write_text(json.dumps(frozen_doctor(ready=False)))
            native = root / "native.json"; native.write_text(json.dumps(native_evidence_template("windows")))
            output = root / "qualification.json"
            code = main(["qualify-release", "--config", str(Path(__file__).resolve().parents[1] / "config.example.json"), "--distribution", str(distribution), "--samples", str(samples), "--frozen-doctor", str(frozen), "--native-evidence", str(native), "--output", str(output)])
            self.assertEqual(code, 2)
            report = json.loads(output.read_text())
            self.assertFalse(report["ready"]); self.assertEqual(report["checks"]["host"]["status"], "blocked"); self.assertEqual(report["checks"]["authenticode"]["status"], "blocked")


if __name__ == "__main__":
    unittest.main()
