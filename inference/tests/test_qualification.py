from __future__ import annotations

import hashlib
import copy
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from jarvis_inference.durable import generic_report_matches, read_report
from jarvis_inference.errors import ApplianceError
from jarvis_inference.manifest import load_manifest
from jarvis_inference.primary_qualification import _PrimaryQualificationProduction
from jarvis_inference.qualification import (
    SUITE_VERSION,
    _qualification_child_environment,
    _QualificationProduction,
    evaluate_evidence,
    qualify,
)
from jarvis_inference.strict_json import canonical_bytes

TEST_HARDWARE_ENV = {
    "JARVIS_QUAL_GPU_DRIVER": "test",
    "JARVIS_QUAL_COMPUTE_RUNTIME": "CUDA 12.8",
}


def passing_report(manifest, identity: dict, runner: dict) -> dict:
    """Synthetic passing evidence for transport and report-admission tests."""
    from jarvis_inference.qualification import (
        _WINDOWS_QUALIFICATION_ASSETS,
        _load_windows_qualification_assets,
    )

    evidence = QualificationTests().evidence(identity)
    evidence["tools"] = (
        {"named_success": 10, "required_success": 10}
        if manifest.capabilities.tools
        else {"named_success": 0, "required_success": 0}
    )
    evidence["json_schema"] = {
        "strict_success": 20 if manifest.capabilities.json_schema else 0
    }
    if manifest.platform == "windows-x86_64":
        for name, (relative, _) in _WINDOWS_QUALIFICATION_ASSETS.items():
            path = manifest.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(name.encode())
        assets = _load_windows_qualification_assets(manifest)
        evidence["platform_gate"]["assets"] = assets
        evidence["platform_gate"]["audio_corpus_digest"] = assets["audio_corpus"][
            "sha256"
        ]
        evidence["fixture_digest"] = hashlib.sha256(
            canonical_bytes({"runner": runner, "assets": assets})
        ).hexdigest()
    else:
        evidence["platform_gate"] = {
            "kind": "mac_unified_memory",
            "warmup_requests": 20,
            "normal_requests": 200,
            "final_drift": 0.01,
            "theil_sen_projected_growth": 0.01,
            "oom": 0,
            "crashes": 0,
            "stale_state_references": 0,
        }
    evidence["provenance"].update(
        identity_digest=hashlib.sha256(canonical_bytes(identity)).hexdigest(),
        fixture_digest=evidence["fixture_digest"],
        started_unix_ns="1100000000",
        completed_unix_ns="1900000000",
        monotonic_elapsed_ns=800000000,
    )
    metadata = {
        "suite_version",
        "fixture_digest",
        "deterministic_seeds",
        "identity",
        "provenance",
    }
    return {
        "report_schema_version": 1,
        "status": "pass",
        "started": 1,
        "completed": 2,
        "evidence_digest": hashlib.sha256(canonical_bytes(evidence)).hexdigest(),
        "fixture_runner": runner,
        **{key: evidence[key] for key in metadata},
        "results": {
            key: value for key, value in evidence.items() if key not in metadata
        },
    }


def artifact(path: Path, root: Path) -> dict:
    raw = path.read_bytes()
    return {"path": path.relative_to(root).as_posix(), "size_bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


class QualificationTests(unittest.TestCase):
    def test_persisted_windows_report_rechecks_gates_and_current_assets(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = self.manifest(root)
            runner = root / "fixture-runner"
            runner.write_bytes(b"runner")
            identity = self.identity()
            good = passing_report(manifest, identity, artifact(runner, root))
            self.assertTrue(generic_report_matches(manifest, identity, good))
            mutations = [
                ("results", {}, None),
                ("results", {"greek_success": 20}, "streaming"),
                (
                    "results",
                    {"greek_success": 20.0, "english_success": 20},
                    "streaming",
                ),
                ("report_schema_version", True, None),
                ("provenance", "0" * 64, "identity_digest"),
                ("provenance", "3000000000", "completed_unix_ns"),
                ("provenance", None, None),
            ]
            for field, value, child in mutations:
                bad = copy.deepcopy(good)
                if child is None:
                    bad[field] = value
                else:
                    bad[field][child] = value
                with self.subTest(field=field, child=child, value=value):
                    self.assertFalse(generic_report_matches(manifest, identity, bad))
            bad = copy.deepcopy(good)
            bad["results"]["platform_gate"]["coexistence_p95"] = 2
            self.assertFalse(generic_report_matches(manifest, identity, bad))
            asset = (
                root
                / good["results"]["platform_gate"]["assets"]["audio_corpus"]["path"]
            )
            asset.write_bytes(b"changed corpus")
            self.assertFalse(generic_report_matches(manifest, identity, good))

    def test_stopped_or_failed_qualification_supervision_is_not_ready(self) -> None:
        for production_type in (_QualificationProduction, _PrimaryQualificationProduction):
            with self.subTest(production=production_type.__name__):
                production = production_type(SimpleNamespace(), "test", None)
                production.engine = SimpleNamespace(ready=lambda: True)
                production.ready_state = True
                production.stop.set()
                self.assertFalse(production.is_ready())
                production.stop.clear()
                failure = OSError("supervision failed")
                with patch.object(production_type, "_supervise_loop", side_effect=failure):
                    production._supervise()
                self.assertFalse(production.is_ready())
                self.assertIs(production.loader_failure, failure)

    def test_generic_and_profile_supervisors_cannot_publish_after_close(self) -> None:
        for production_type in (
            _QualificationProduction,
            _PrimaryQualificationProduction,
        ):
            with self.subTest(production=production_type.__name__):
                factory_entered = threading.Event()
                release_factory = threading.Event()
                candidate_closed = threading.Event()

                class RaceEngine:
                    process = None

                    def __init__(self, ready: bool) -> None:
                        self.open = ready

                    def ready(self) -> bool:
                        return self.open

                    def start(self, deadline=None) -> None:
                        self.open = True

                    def close(self) -> None:
                        self.open = False
                        if self is candidate:
                            candidate_closed.set()

                candidate = RaceEngine(False)

                def factory(_manifest):
                    factory_entered.set()
                    release_factory.wait(5)
                    return candidate

                production = production_type(
                    SimpleNamespace(),
                    "k" * 40,
                    factory,
                    engine=RaceEngine(False),
                )
                production.start_background()
                self.assertTrue(factory_entered.wait(2))
                failures = []

                def close_production() -> None:
                    try:
                        production.close()
                    except BaseException as exc:  # noqa: BLE001 - test capture
                        failures.append(exc)

                closer = threading.Thread(target=close_production)
                closer.start()
                stop = time.monotonic() + 2
                while not production.stop.is_set() and time.monotonic() < stop:
                    time.sleep(0.01)
                release_factory.set()
                closer.join(timeout=3)
                self.assertFalse(closer.is_alive())
                self.assertEqual(failures, [])
                self.assertTrue(candidate_closed.is_set())
                self.assertIsNone(production.engine)
                self.assertIsNone(production.loading_engine)
                production.require_teardown_proof()

    def test_close_redrains_late_engine_before_teardown_proof(self) -> None:
        closed = []

        class Engine:
            def __init__(self, name: str) -> None:
                self.name = name

            def ready(self) -> bool:
                return True

            def close(self) -> None:
                closed.append(self.name)

        production = _QualificationProduction(
            SimpleNamespace(), "k" * 40, lambda _: None, engine=Engine("initial")
        )
        late_active = Engine("late-active")
        late_loading = Engine("late-loading")

        class Loader:
            def join(self, timeout=None) -> None:
                with production.state_lock:
                    production.engine = late_active
                    production.loading_engine = late_loading
                production.loader_exited.set()

            def is_alive(self) -> bool:
                return False

        production.loader = Loader()
        production.close()
        production.require_teardown_proof()
        self.assertEqual(
            closed, ["initial", "late-active", "late-loading"]
        )

    def test_sealed_fixture_runner_does_not_inherit_path(self) -> None:
        child = _qualification_child_environment(
            {
                "PATH": "attacker-controlled",
                "SYSTEMROOT": r"C:\Windows",
                "TEMP": r"C:\Windows\Temp",
                "TMP": r"C:\ambient-tmp",
                "TMPDIR": r"C:\ambient-tmpdir",
            }
        )
        self.assertNotIn("PATH", child)
        self.assertNotIn("SYSTEMROOT", child)
        self.assertNotIn("WINDIR", child)
        self.assertNotIn("TEMP", child)
        self.assertNotIn("TMP", child)
        self.assertNotIn("TMPDIR", child)
        with tempfile.TemporaryDirectory(
            dir=Path(tempfile.gettempdir()).resolve()
        ) as directory:
            trusted = Path(directory)
            private = _qualification_child_environment(
                {"TEMP": "ambient", "SYSTEMROOT": r"C:\Windows"},
                trusted,
            )
            self.assertEqual(private["TEMP"], str(trusted))
            self.assertEqual(private["TMP"], str(trusted))
            self.assertEqual(private["TMPDIR"], str(trusted))

    @staticmethod
    def identity() -> dict:
        return {"id": "exact", "hardware_runtime": {"driver": "test", "compute_runtime": "CUDA 12.8"}}

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
            "windows_assets": self.windows_assets(),
        }

    @staticmethod
    def windows_assets() -> dict:
        return {
            name: {"path": f"qualification/{name}.bin", "size_bytes": 1, "sha256": character * 64}
            for name, character in (
                ("fixture", "a"),
                ("collector", "b"),
                ("audio_corpus", "c"),
                ("power_profile", "d"),
                ("asr_config", "e"),
            )
        }

    def evidence(self, identity: dict) -> dict:
        power = {
            "windows_power_scheme_guid": "12345678-1234-1234-1234-123456789abc",
            "ac_state": "ac", "gpu_power_limit_watts": 360,
            "gpu_clock_policy": "applications_graphics_mhz=2500;applications_memory_mhz=14000",
            "driver_version": "test", "compute_runtime": "CUDA 12.8",
            "require_zero_thermal_power_throttle_reasons": True,
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
                "assets": self.windows_assets(),
                "asr_runtime_identity_digest": "f" * 64,
                "fixture": {
                    "fixture_schema_version": 1,
                    "fixture_id": "cyrus-windows-asr-coexistence-v1",
                    "platform": "windows-x86_64",
                    "prompt_workload_units": [512, 2048, 4096],
                    "max_completion_tokens": 512, "request_count": 100, "concurrency": 1,
                    "schedule": "serial-continuous-from-audio-start", "minimum_overlapping_utterances": 80,
                    "sampling": {"temperature": 0, "top_p": 1},
                    "audio_corpus": {
                        "schema": "cyrus.windows-asr-audio-corpus/1",
                        "utterance_count": 200,
                        "language_counts": {"el": 100, "en": 100},
                        "sample_rate_hz": 16000,
                        "channels": 1,
                        "sample_width_bytes": 2,
                        "minimum_duration_ms": 1500,
                        "maximum_duration_ms": 30000,
                        "clean_language_accuracy_minimum_correct_per_language": 95,
                        "clean_aggregate_wer_max_percent_per_language": 20,
                        "coexistence_selection": "first-50-per-locale",
                        "reference_sidecars": {
                            "encoding": "utf-8",
                            "normalization": "NFC",
                            "controls_allowed": False,
                            "word_normalization": "unicode-lmn-casefold-v1",
                        },
                    },
                    "power_profile": power,
                },
                "audio_corpus_digest": "c" * 64,
                "greek_utterances": 100,
                "english_utterances": 100,
                "greek_language_correct": 95,
                "english_language_correct": 95,
                "greek_reference_words": 500,
                "greek_word_errors": 100,
                "english_reference_words": 500,
                "english_word_errors": 100,
                "forced_cpu_fallbacks": 5,
                "baseline_power_identity_digest": power_digest, "coexistence_power_identity_digest": power_digest,
                "overlapping_utterances": 80, "baseline_p95": 1.0, "coexistence_p95": 1.15,
                "baseline_llm_p95": 10.0, "coexistence_llm_p95": 11.5,
                "lost_final_transcripts": 0, "duplicate_final_transcripts": 0,
                "stale_results": 0, "oom": 0, "crashes": 0, "throttle_reasons": 0,
                "minimum_free_vram_mib": 2048,
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

    def test_language_wer_llm_fallback_and_result_integrity_gates_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest = self.manifest(Path(temporary))
            identity = self.identity()
            mutations = (
                ("english_language_correct", 94),
                ("greek_word_errors", 101),
                ("coexistence_llm_p95", 11.501),
                ("forced_cpu_fallbacks", 4),
                ("duplicate_final_transcripts", 1),
                ("stale_results", 1),
            )
            for field, value in mutations:
                evidence = self.evidence(identity)
                evidence["platform_gate"][field] = value
                with self.subTest(field=field), self.assertRaises(ApplianceError):
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

    def test_qualification_cancel_uses_forced_injection_not_cooperative_cancel(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest = self.manifest(Path(temporary))

            class Engine:
                def __init__(self) -> None:
                    self.process = SimpleNamespace(pid=123)
                    self.injected = 0
                    self.phase = "prefill"

                def ready(self) -> bool:
                    return True

                def inject_cancel(self) -> None:
                    self.injected += 1
                    self.phase = "idle"

                def cancel(self) -> None:
                    raise AssertionError("cooperative cancel must not be used")

                def runtime_snapshot(self) -> dict:
                    return {
                        "phase": self.phase,
                        "pid": 123,
                        "termination_serial": self.injected,
                        "last_termination_kind": "cancel",
                        "last_termination_phase": "prefill",
                        "last_termination_elapsed": 0.01,
                        "last_termination_exit_ok": True,
                    }

            engine = Engine()
            production = _QualificationProduction(
                manifest,
                "secret",
                lambda _manifest: engine,
                engine=engine,
            )
            result = production.inject_qualification_cancel("prefill")
            self.assertEqual(engine.injected, 1)
            self.assertEqual(result["phase"], "prefill")
            self.assertTrue(result["exit_ok"])
