from __future__ import annotations

import hashlib
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from jarvis_inference.primary_profile import load_primary_corpus
from jarvis_inference.strict_json import canonical_bytes, parse_body
from jarvis_inference.windows_primary_fixture import load_fixture
from jarvis_inference.windows_primary_runner import (
    PrimaryRunnerFailure,
    _attach_cold_load_observations,
    _checked_usage,
    _safe_environment,
    _write_exclusive,
    run_performance,
    run_quality,
    run_soak,
)
from jarvis_inference.windows_primary_metrics import evaluate_memory_epochs


ROOT = Path(__file__).parents[1]
REPOSITORY = ROOT.parent
QUALITY = ROOT / "fixtures" / "local-primary-quality-v1.json"
PERFORMANCE = ROOT / "fixtures" / "windows-primary-performance-soak-v1.json"


class QualityClient:
    def __init__(self, failed: set[str] | None = None, critical: set[str] | None = None):
        self.failed = failed or set()
        self.critical = critical or set()
        self.calls: list[str] = []

    def readiness(self) -> None:
        return None

    def capabilities(self) -> None:
        return None

    def stream(self, fixture):
        self.calls.append(fixture.case_id)
        return {
            "case_id": fixture.case_id,
            "protocol_pass": True,
            "quality_pass": fixture.case_id not in self.failed,
            "forbidden_content_ok": fixture.case_id not in self.critical,
        }


class PerformanceClient:
    def __init__(self, targets: list[int]):
        self.targets = targets
        self.index = 0

    def tokenize(self, request):
        count = self.targets[self.index]
        return {
            "token_count": count,
            "token_ids_sha256": hashlib.sha256(
                f"tokens-{self.index}-{count}".encode("ascii")
            ).hexdigest(),
        }

    def observe(self, request, expected_tokenization):
        serial = self.index + 1
        self.index += 1
        return {
            "generation_serial": serial,
            "ttft_seconds": 0.5,
            "end_to_end_seconds": 5.0,
            "accepted_decode_tokens": 128,
            "accepted_token_ids_sha256": hashlib.sha256(
                f"accepted-{serial}".encode("ascii")
            ).hexdigest(),
            "backend_prompt_seconds": 0.5,
            "backend_decode_seconds": 4.0,
            "decode_tokens_per_second": 32.0,
            "request_to_engine_final_seconds": 4.5,
            "finish_reason": "tool_calls" if request.get("tools") else "length",
        }


class FakeClock:
    def __init__(self):
        self.value = 1.0

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += max(seconds, 0.000001)


class SoakClient:
    def __init__(self, clock: FakeClock):
        self.clock = clock
        self.pid = 100
        self.observations = 0
        self.restarts = 0
        self.terminations: list[dict[str, object]] = []

    def wait_ready(self, *, sleep):
        return self.runtime()

    def observe(self, request, expected_tokenization):
        self.observations += 1
        self.clock.advance(0.00001)
        return {}

    def runtime(self):
        return {
            "ready": True,
            "busy": False,
            "gateway_pid": 50,
            "engine": {"pid": self.pid},
            "terminations": list(self.terminations),
        }

    def restart(self, previous_pid):
        self.restarts += 1
        self.terminations.append(
            {
                "pid": previous_pid,
                "kind": "crash",
                "phase": "idle",
                "elapsed_seconds": 0.001,
                "exit_ok": True,
            }
        )
        self.pid += 1
        self.clock.advance(0.01)
        return self.runtime(), 0.01


def _manifest() -> SimpleNamespace:
    return SimpleNamespace(
        digest="a" * 64,
        backend={"llama_cpp_commit": "b" * 40, "build_id": "native-build"},
        model={"artifacts": [{"sha256": "c" * 64}]},
        prompt_profile={
            "id": "google-gemma4-12b-it-qat-q4_0-v1",
            "renderer": {"sha256": "d" * 64},
        },
    )


class WindowsPrimaryRunnerTests(unittest.TestCase):
    def test_quality_executes_all_120_cases_and_keeps_only_verdicts(self) -> None:
        fixtures, _ = load_primary_corpus(QUALITY)
        client = QualityClient(failed={"el_instruction_01", "en_instruction_01"})
        report = run_quality(client, fixtures)

        self.assertEqual(len(client.calls), 122)
        self.assertEqual(report["sample_count"], 120)
        self.assertEqual(report["greek_passed"], 59)
        self.assertEqual(report["english_passed"], 59)
        self.assertEqual(
            set(report["cases"][0]), {"case_id", "passed", "critical"}
        )

    def test_quality_fails_closed_on_threshold_or_critical_failure(self) -> None:
        fixtures, _ = load_primary_corpus(QUALITY)
        greek = {fixture.case_id for fixture in fixtures if fixture.language == "el"}
        with self.assertRaisesRegex(PrimaryRunnerFailure, "thresholds"):
            run_quality(QualityClient(failed=set(sorted(greek)[:7])), fixtures)
        with self.assertRaisesRegex(PrimaryRunnerFailure, "thresholds"):
            run_quality(QualityClient(critical={fixtures[0].case_id}), fixtures)

    def test_performance_binds_all_native_token_observations(self) -> None:
        fixture = load_fixture(PERFORMANCE)
        targets = [case["target_rendered_bin"] for case in fixture.document["performance_cases"]]
        report, tokenizer_sha, tokenizations = run_performance(
            PerformanceClient(targets),
            fixture,
            runtime_identity_sha256="e" * 64,
            manifest=_manifest(),
        )

        self.assertEqual(len(report["cases"]), 100)
        self.assertEqual(
            tuple(item["token_count"] for item in tokenizations), tuple(targets)
        )
        self.assertEqual(report["native_token_validation"]["validated_cases"], 100)
        self.assertEqual(report["cold_load_to_ready_samples_seconds"], [])
        self.assertEqual(report["native_token_validation"]["tokenizer_identity_sha256"], tokenizer_sha)
        self.assertEqual(
            tokenizer_sha,
            "71210d0a4f8d78966c59e5cae3547753e1d87ae8be28ab76a2feb30aaa6c7c30",
        )
        serialized = canonical_bytes(report).decode("utf-8")
        self.assertNotIn('"messages"', serialized)
        self.assertNotIn('"content"', serialized)
        self.assertNotIn('"token_ids"', serialized)

    def test_token_accounting_fails_closed(self) -> None:
        self.assertEqual(
            _checked_usage(
                {"prompt_tokens": 8, "completion_tokens": 4, "total_tokens": 12},
                8,
            ),
            4,
        )
        with self.assertRaises(PrimaryRunnerFailure):
            _checked_usage(
                {"prompt_tokens": 7, "completion_tokens": 4, "total_tokens": 11},
                8,
            )
    def test_short_soak_runs_every_case_every_cycle_and_restarts(self) -> None:
        clock = FakeClock()
        client = SoakClient(clock)
        requests = tuple({"case": index} for index in range(100))
        soak, epochs, cold = run_soak(
            client,
            requests,
            tuple(10 for _ in requests),
            minimum_seconds=0.08,
            cycles=8,
            restart_every=4,
            minimum_epoch_samples=4,
            memory_sampler=lambda gateway_pid, engine_pid: 1_000_000,
            clock=clock,
            sleep=clock.advance,
        )

        self.assertEqual(client.observations, 800)
        self.assertEqual(soak["cycles_completed"], 8)
        self.assertEqual(soak["full_restarts_completed"], 2)
        self.assertEqual(epochs, [1, 2, 3])
        self.assertEqual(cold, [0.01, 0.01])
        self.assertTrue(all(value == 0 for value in soak["event_counts"].values()))
        self.assertEqual(soak["memory"]["epoch_sample_counts"], [4, 4])

    def test_cold_load_samples_are_measured_exactly_and_recomputed(self) -> None:
        performance = {
            "cold_load_to_ready_samples_seconds": [],
            "metrics": {},
        }
        samples = [float(value) for value in range(1, 9)]
        _attach_cold_load_observations(performance, samples)
        self.assertEqual(
            performance["cold_load_to_ready_samples_seconds"], samples
        )
        self.assertAlmostEqual(
            performance["metrics"]["cold_load_to_ready_p95_seconds"], 7.65
        )
        with self.assertRaisesRegex(PrimaryRunnerFailure, "sample set"):
            _attach_cold_load_observations(
                {"cold_load_to_ready_samples_seconds": [], "metrics": {}},
                samples[:-1],
            )
        with self.assertRaisesRegex(PrimaryRunnerFailure, "threshold"):
            _attach_cold_load_observations(
                {"cold_load_to_ready_samples_seconds": [], "metrics": {}},
                [121.0] * 8,
            )

    def test_memory_transient_spike_passes_but_restart_sawtooth_leak_fails(self) -> None:
        transient = [[100, 100, 100, 180, 100, 100, 100, 100, 100, 100, 100, 100]]
        result = evaluate_memory_epochs(transient)
        self.assertEqual(result[:3], (100, 100, 180))
        self.assertFalse(result[-1])

        hidden_by_restart = [
            [100, 100, 101, 102, 103, 104, 105, 106, 107, 108, 110, 112],
            [100, 100, 101, 102, 103, 104, 105, 106, 107, 108, 110, 112],
        ]
        leaking = evaluate_memory_epochs(hidden_by_restart)
        self.assertGreater(leaking[4], 0.05)
        self.assertTrue(leaking[-1])

    def test_environment_is_env_only_and_rejects_existing_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            values = {
                "CYRUS_PROFILE_QUAL_CHALLENGE": "a" * 64,
                "CYRUS_PROFILE_QUAL_OUTPUT": str(root / "evidence.json"),
                "CYRUS_PROFILE_QUAL_DIAGNOSTIC": str(root / "diagnostic.json"),
                "CYRUS_PROFILE_QUAL_MANIFEST": str(root / "manifest.json"),
                "CYRUS_PROFILE_QUAL_BASE_URL": "http://127.0.0.1:8766/v1",
                "CYRUS_PROFILE_QUAL_API_KEY": "k" * 48,
            }
            with patch.dict(os.environ, values, clear=True):
                self.assertEqual(_safe_environment()[0], "a" * 64)
                (root / "evidence.json").write_bytes(b"old")
                with self.assertRaises(PrimaryRunnerFailure):
                    _safe_environment()

    def test_atomic_writer_never_replaces_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "evidence.json"
            document = {"status": "pass", "value": 1}
            _write_exclusive(output, document)
            self.assertEqual(parse_body(output.read_bytes()), document)
            with self.assertRaises((FileExistsError, PrimaryRunnerFailure)):
                _write_exclusive(output, {"status": "changed"})
            self.assertEqual(parse_body(output.read_bytes()), document)

    def test_native_build_contract_is_one_file_and_integrated(self) -> None:
        spec = (REPOSITORY / "packaging" / "windows_primary_profile_runner.spec").read_text(encoding="utf-8")
        generic_spec = (ROOT / "packaging" / "windows_native_fixture_runner.spec").read_text(encoding="utf-8")
        runtime_hook = (ROOT / "packaging" / "runtime_hook_harden_dll.py").read_text(encoding="utf-8")
        build = (REPOSITORY / "scripts" / "build_windows.ps1").read_text(encoding="utf-8")
        entry = (ROOT / "scripts" / "windows_primary_profile_runner.py").read_text(encoding="utf-8")

        self.assertIn('name="cyrus-windows-primary-profile-runner"', spec)
        self.assertIn("analysis.binaries", spec)
        self.assertNotIn("COLLECT(", spec)
        self.assertIn("runtime_hook_harden_dll.py", spec)
        self.assertIn("runtime_hook_harden_dll.py", generic_spec)
        self.assertIn("analysis.binaries", generic_spec)
        self.assertIn("analysis.datas", generic_spec)
        self.assertNotIn("exclude_binaries=True", generic_spec)
        self.assertNotIn("COLLECT(", generic_spec)
        self.assertNotIn("jarvis_voice", runtime_hook)
        self.assertIn("SetDefaultDllDirectories", runtime_hook)
        self.assertIn("os.add_dll_directory", runtime_hook)
        self.assertIn("windows_primary_profile_runner.spec", build)
        self.assertIn(
            '$NativeFixtureRunner = Join-Path $RunnerOutputDirectory "native-fixture-runner.exe"',
            build,
        )
        self.assertNotIn(
            "native-fixture-runner\\native-fixture-runner.exe", build
        )
        self.assertIn('(Join-Path $RunnerWorkDirectory "primary-profile")', build)
        self.assertIn(
            '$PrimaryProfileRunner = Join-Path $RunnerOutputDirectory '
            '"cyrus-windows-primary-profile-runner.exe"',
            build,
        )
        self.assertIn("cyrus-windows-primary-profile-runner.exe", build)
        self.assertIn("jarvis_inference.windows_primary_runner import main", entry)


if __name__ == "__main__":
    unittest.main()
