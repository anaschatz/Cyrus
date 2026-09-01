"""Native Windows primary-profile qualification fixture runner.

This module is bundled as one console runner, hash-bound executable.  It talks
only to the ephemeral loopback gateway created by ``primary_qualification``.
Prompts and model outputs exist only in process memory; the sole durable output
is a challenge-bound document containing identities, booleans, counters and
timings.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
import stat
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

from .manifest import hash_file, load_manifest, resolve_local
from .primary_profile import (
    PRIMARY_CORPUS_SHA256,
    PRIMARY_QUALITY_SUITE_VERSION,
    PRIMARY_SYSTEM_PROMPT,
    load_primary_corpus,
)
from .primary_quality_client import PrimaryQualityClient
from .profile_preflight import Fixture, PreflightFailure, percentile
from .profile_report import (
    ATTESTATION_LEVEL,
    EVIDENCE_SCHEMA,
    OPERATOR_TRUST_REQUIRED,
    PERFORMANCE_FIXTURE_RELATIVE,
    PRIMARY_CORPUS_RELATIVE,
    PROFILE_RUNNER_RELATIVE,
    expected_profile_identity,
)
from .strict_json import canonical_bytes, parse_body
from .windows_primary_fixture import LoadedWindowsPrimaryFixture, load_fixture
from .windows_primary_metrics import (
    PrimaryQualificationClient,
    PrimaryRunnerFailure,
    _SHA,
    _checked_usage,
    _exact,
    combined_working_set_bytes,
    evaluate_memory_epochs,
)


_CHALLENGE = re.compile(r"^[0-9a-f]{64}$", re.ASCII)
_STAGES = {
    "bootstrap",
    "quality",
    "performance",
    "soak",
    "evidence_write",
}
_ZERO_EVENTS = (
    "observed_request_failures",
    "observed_unexpected_process_terminations",
    "observed_request_timeouts",
    "observed_cleanup_failures",
    "observed_circuit_breaker_rejections",
)
_current_stage = "bootstrap"


def _stage(value: str) -> None:
    global _current_stage
    if value not in _STAGES:
        raise PrimaryRunnerFailure("invalid runner stage")
    _current_stage = value


def run_quality(
    client: Any,
    fixtures: tuple[Fixture, ...],
    *,
    warmups: int = 2,
) -> dict[str, Any]:
    """Run all 120 cases and retain only deterministic verdicts."""

    if len(fixtures) != 120 or warmups != 2:
        raise PrimaryRunnerFailure("quality fixture count is invalid")
    client.readiness()
    client.capabilities()
    warmup = Fixture(
        "warmup", "en", "Reply with only this word: ready", 16, exact="ready"
    )
    for _ in range(warmups):
        client.stream(warmup)
    cases: list[dict[str, Any]] = []
    language_passes = {"el": 0, "en": 0}
    critical_failures = 0
    for fixture in fixtures:
        try:
            observation = client.stream(fixture)
        except (OSError, PreflightFailure) as exc:
            raise PrimaryRunnerFailure("quality protocol case failed") from exc
        if (
            observation.get("case_id") != fixture.case_id
            or observation.get("protocol_pass") is not True
            or type(observation.get("quality_pass")) is not bool
            or type(observation.get("forbidden_content_ok")) is not bool
        ):
            raise PrimaryRunnerFailure("quality observation is invalid")
        passed = observation["quality_pass"]
        critical = not observation["forbidden_content_ok"]
        language_passes[fixture.language] += int(passed)
        critical_failures += int(critical)
        cases.append(
            {"case_id": fixture.case_id, "passed": passed, "critical": critical}
        )
    gap = abs(language_passes["el"] / 60 - language_passes["en"] / 60)
    if (
        language_passes["el"] < 54
        or language_passes["en"] < 54
        or gap > 0.10
        or critical_failures != 0
    ):
        raise PrimaryRunnerFailure("quality thresholds failed")
    return {
        "status": "pass",
        "suite_version": PRIMARY_QUALITY_SUITE_VERSION,
        "corpus_sha256": PRIMARY_CORPUS_SHA256,
        "system_prompt_sha256": hashlib.sha256(
            PRIMARY_SYSTEM_PROMPT.encode("utf-8")
        ).hexdigest(),
        "sample_count": 120,
        "greek_total": 60,
        "greek_passed": language_passes["el"],
        "english_total": 60,
        "english_passed": language_passes["en"],
        "language_pass_rate_gap": gap,
        "critical_failures": critical_failures,
        "protocol_failures": 0,
        "cases": cases,
    }


def run_performance(
    client: Any,
    fixture: LoadedWindowsPrimaryFixture,
    *,
    runtime_identity_sha256: str,
    manifest: Any,
) -> tuple[dict[str, Any], str, tuple[dict[str, Any], ...]]:
    """Run the frozen requests once and bind native tokenizer observations."""

    if len(fixture.requests) != 100 or _SHA.fullmatch(runtime_identity_sha256) is None:
        raise PrimaryRunnerFailure("performance fixture identity is invalid")
    cases: list[dict[str, Any]] = []
    tokenizer_rows: list[dict[str, Any]] = []
    tokenizations: list[dict[str, Any]] = []
    for contract, request in zip(
        fixture.document["performance_cases"], fixture.requests, strict=True
    ):
        tokenization = client.tokenize(request)
        prompt_count = tokenization["token_count"]
        target_range = fixture.document["execution"]["native_token_count_ranges"][
            str(contract["target_rendered_bin"])
        ]
        if not target_range["minimum"] <= prompt_count <= target_range["maximum"]:
            raise PrimaryRunnerFailure("native prompt token count missed its frozen bin")
        observation = client.observe(request, tokenization)
        projected_128 = observation["ttft_seconds"] + 127 / observation[
            "decode_tokens_per_second"
        ]
        row = {
            "case_id": contract["case_id"],
            "target_bin": contract["target_rendered_bin"],
            "rendered_token_count": prompt_count,
            "rendered_token_ids_sha256": tokenization["token_ids_sha256"],
            **observation,
            "projected_128_token_e2e_seconds": projected_128,
            "passed": True,
        }
        cases.append(row)
        tokenizations.append(dict(tokenization))
        tokenizer_rows.append(
            {
                "case_id": contract["case_id"],
                "rendered_token_count": prompt_count,
                "rendered_token_ids_sha256": tokenization["token_ids_sha256"],
            }
        )
    ttft = [float(case["ttft_seconds"]) for case in cases]
    end_to_end = [float(case["end_to_end_seconds"]) for case in cases]
    decode = [float(case["decode_tokens_per_second"]) for case in cases]
    projected = [
        float(case["projected_128_token_e2e_seconds"]) for case in cases
    ]
    metrics = {
        "ttft_p50_seconds": percentile(ttft, 0.50),
        "ttft_p95_seconds": percentile(ttft, 0.95),
        "end_to_end_p50_seconds": percentile(end_to_end, 0.50),
        "end_to_end_p95_seconds": percentile(end_to_end, 0.95),
        "projected_128_token_e2e_p95_seconds": percentile(projected, 0.95),
        "decode_p5_tokens_per_second": percentile(decode, 0.05),
        "decode_p50_tokens_per_second": percentile(decode, 0.50),
        "decode_p95_tokens_per_second": percentile(decode, 0.95),
    }
    if (
        metrics["ttft_p95_seconds"] > 3.0
        or metrics["projected_128_token_e2e_p95_seconds"] > 10.0
        or metrics["decode_p5_tokens_per_second"] < 25.0
    ):
        raise PrimaryRunnerFailure("performance thresholds failed")
    artifact = manifest.model["artifacts"][0]
    tokenizer_identity = {
        "schema": "cyrus.windows-native-tokenizer-identity/1",
        "manifest_sha256": manifest.digest,
        "runtime_identity_sha256": runtime_identity_sha256,
        "backend": {
            "llama_cpp_commit": manifest.backend["llama_cpp_commit"],
            "build_id": manifest.backend["build_id"],
        },
        "model_artifact_sha256": artifact["sha256"],
        "prompt_profile_id": manifest.prompt_profile["id"],
        "renderer_sha256": manifest.prompt_profile["renderer"]["sha256"],
        "cases": tokenizer_rows,
    }
    tokenizer_sha = hashlib.sha256(canonical_bytes(tokenizer_identity)).hexdigest()
    performance = {
        "status": "pass",
        "suite_id": fixture.document["suite_id"],
        "fixture_sha256": fixture.fixture_sha256,
        "expanded_requests_sha256": fixture.expanded_requests_sha256,
        "rendered_prompt_texts_sha256": fixture.rendered_prompt_texts_sha256,
        "timed_requests_completed": 100,
        # Populated only after the later native soak has observed every
        # deliberate full restart reach ready. An empty list cannot validate
        # as pass evidence.
        "cold_load_to_ready_samples_seconds": [],
        "native_token_validation": {
            "status": "pass",
            "validated_cases": 100,
            "tokenizer_identity_sha256": tokenizer_sha,
        },
        "metrics": metrics,
        "cases": cases,
    }
    serials = [case["generation_serial"] for case in cases]
    if (
        any(type(serial) is not int or serial <= 0 for serial in serials)
        or serials != sorted(set(serials))
    ):
        raise PrimaryRunnerFailure("performance generation serials are invalid")
    return performance, tokenizer_sha, tuple(tokenizations)


def _engine_pid(runtime: dict[str, Any]) -> int:
    engine = runtime.get("engine")
    pid = engine.get("pid") if isinstance(engine, dict) else None
    if type(pid) is not int or pid <= 0:
        raise PrimaryRunnerFailure("managed engine pid is unavailable")
    return pid


def _attach_cold_load_observations(
    performance: dict[str, Any], samples: list[float]
) -> None:
    """Attach exactly eight native restart-to-ready observations."""

    if (
        len(samples) != 8
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= 0
            for value in samples
        )
        or not isinstance(performance.get("metrics"), dict)
        or performance.get("cold_load_to_ready_samples_seconds") != []
    ):
        raise PrimaryRunnerFailure("cold-load sample set is incomplete")
    normalized = [float(value) for value in samples]
    cold_p95 = percentile(normalized, 0.95)
    if cold_p95 > 120.0:
        raise PrimaryRunnerFailure("cold-load threshold failed")
    performance["cold_load_to_ready_samples_seconds"] = normalized
    performance["metrics"]["cold_load_to_ready_p95_seconds"] = cold_p95


def run_soak(
    client: Any,
    requests: tuple[dict[str, Any], ...],
    tokenizations: tuple[dict[str, Any], ...],
    *,
    minimum_seconds: float,
    cycles: int,
    restart_every: int,
    minimum_epoch_samples: int = 12,
    memory_sampler: Callable[[int, int], int] = combined_working_set_bytes,
    clock: Callable[[], float] = time.perf_counter,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[dict[str, Any], list[int], list[float]]:
    """Exercise every frozen request per cycle for a monotonic soak window."""

    if (
        len(requests) != 100
        or len(tokenizations) != 100
        or not math.isfinite(minimum_seconds)
        or minimum_seconds <= 0
        or type(cycles) is not int
        or cycles < 1
        or type(restart_every) is not int
        or restart_every < 1
        or restart_every > cycles
        or cycles % restart_every != 0
        or type(minimum_epoch_samples) is not int
        or minimum_epoch_samples < 2
        or restart_every < minimum_epoch_samples
    ):
        raise PrimaryRunnerFailure("invalid soak contract")
    initial = client.wait_ready(sleep=sleep)
    pid = _engine_pid(initial)
    gateway_pid = initial.get("gateway_pid")
    if type(gateway_pid) is not int or gateway_pid <= 0 or gateway_pid == pid:
        raise PrimaryRunnerFailure("qualification gateway pid is unavailable")
    if initial.get("terminations") != []:
        raise PrimaryRunnerFailure("soak began with unexpected process terminations")
    restart_epochs = [1]
    restart_durations: list[float] = []
    memory_epochs: list[list[int]] = [[]]
    expected_terminations = 0
    started = clock()
    for cycle in range(1, cycles + 1):
        for request, tokenization in zip(requests, tokenizations, strict=True):
            client.observe(request, tokenization)
        runtime = client.runtime()
        terminations = runtime.get("terminations")
        if (
            runtime.get("ready") is not True
            or runtime.get("busy") is not False
            or runtime.get("gateway_pid") != gateway_pid
            or _engine_pid(runtime) != pid
            or not isinstance(terminations, list)
            or len(terminations) != expected_terminations
            or any(
                item.get("kind") != "crash" or item.get("exit_ok") is not True
                for item in terminations
                if isinstance(item, dict)
            )
        ):
            raise PrimaryRunnerFailure("soak cycle did not return to idle")
        memory_epochs[-1].append(memory_sampler(gateway_pid, pid))
        if cycle % restart_every == 0:
            runtime, duration = client.restart(pid)
            new_pid = _engine_pid(runtime)
            expected_terminations += 1
            if (
                new_pid == pid
                or runtime.get("gateway_pid") != gateway_pid
                or len(runtime.get("terminations", [])) != expected_terminations
                or runtime["terminations"][-1].get("kind") != "crash"
                or not math.isfinite(duration)
                or duration <= 0
            ):
                raise PrimaryRunnerFailure("soak restart evidence is invalid")
            pid = new_pid
            restart_durations.append(duration)
            restart_epochs.append(len(restart_epochs) + 1)
            if cycle < cycles:
                memory_epochs.append([])
        target = started + minimum_seconds * cycle / cycles
        while clock() < target:
            heartbeat = client.runtime()
            if (
                heartbeat.get("ready") is not True
                or heartbeat.get("busy") is not False
                or heartbeat.get("gateway_pid") != gateway_pid
                or _engine_pid(heartbeat) != pid
                or len(heartbeat.get("terminations", []))
                != expected_terminations
            ):
                raise PrimaryRunnerFailure("soak heartbeat lost readiness")
            sleep(min(5.0, max(0.001, target - clock())))
    elapsed = clock() - started
    if elapsed < minimum_seconds:
        raise PrimaryRunnerFailure("soak monotonic duration is incomplete")
    (
        baseline,
        final,
        peak,
        global_growth,
        maximum_epoch_growth,
        epoch_sample_counts,
        growth_detected,
    ) = evaluate_memory_epochs(
        memory_epochs, minimum_samples=minimum_epoch_samples
    )
    if growth_detected:
        raise PrimaryRunnerFailure("soak memory growth gate failed")
    return (
        {
            "status": "pass",
            "monotonic_seconds_observed": elapsed,
            "cycles_completed": cycles,
            "full_restarts_completed": len(restart_durations),
            "event_counts": {name: 0 for name in _ZERO_EVENTS},
            "memory": {
                "evaluated": True,
                "monotonic_growth_detected": False,
                "baseline_bytes": baseline,
                "final_bytes": final,
                "peak_bytes": peak,
                "global_growth_fraction": global_growth,
                "max_epoch_growth_fraction": maximum_epoch_growth,
                "epoch_sample_counts": epoch_sample_counts,
            },
        },
        restart_epochs,
        restart_durations,
    )


def _coexistence_from_generic(generic_report: dict[str, Any]) -> dict[str, Any]:
    try:
        gate = generic_report["results"]["platform_gate"]
    except (KeyError, TypeError) as exc:
        raise PrimaryRunnerFailure("generic coexistence evidence is missing") from exc
    required = {
        "kind",
        "fixture",
        "assets",
        "asr_runtime_identity_digest",
        "audio_corpus_digest",
        "greek_utterances",
        "english_utterances",
        "greek_language_correct",
        "english_language_correct",
        "greek_reference_words",
        "greek_word_errors",
        "english_reference_words",
        "english_word_errors",
        "forced_cpu_fallbacks",
        "baseline_power_identity_digest",
        "coexistence_power_identity_digest",
        "overlapping_utterances",
        "baseline_p95",
        "coexistence_p95",
        "baseline_llm_p95",
        "coexistence_llm_p95",
        "minimum_free_vram_mib",
        "lost_final_transcripts",
        "duplicate_final_transcripts",
        "stale_results",
        "oom",
        "crashes",
        "throttle_reasons",
    }
    gate = _exact(gate, required, "generic coexistence")
    baseline = float(gate["baseline_p95"])
    concurrent = float(gate["coexistence_p95"])
    baseline_llm = float(gate["baseline_llm_p95"])
    concurrent_llm = float(gate["coexistence_llm_p95"])
    if not all(
        math.isfinite(value) and value > 0
        for value in (baseline, concurrent, baseline_llm, concurrent_llm)
    ):
        raise PrimaryRunnerFailure("generic coexistence latency is invalid")
    return {
        "status": "pass",
        "greek_utterances": gate["greek_utterances"],
        "english_utterances": gate["english_utterances"],
        "greek_language_correct": gate["greek_language_correct"],
        "english_language_correct": gate["english_language_correct"],
        "greek_reference_words": gate["greek_reference_words"],
        "greek_word_errors": gate["greek_word_errors"],
        "english_reference_words": gate["english_reference_words"],
        "english_word_errors": gate["english_word_errors"],
        "forced_cpu_fallbacks": gate["forced_cpu_fallbacks"],
        "overlapping_utterances": gate["overlapping_utterances"],
        "baseline_p95_seconds": baseline,
        "coexistence_p95_seconds": concurrent,
        "asr_degradation_ratio": concurrent / baseline - 1,
        "baseline_llm_p95_seconds": baseline_llm,
        "coexistence_llm_p95_seconds": concurrent_llm,
        "llm_regression_ratio": concurrent_llm / baseline_llm - 1,
        "minimum_free_vram_mib": gate["minimum_free_vram_mib"],
        "asr_runtime_identity_sha256": gate["asr_runtime_identity_digest"],
        "lost_final_transcripts": gate["lost_final_transcripts"],
        "duplicate_final_transcripts": gate["duplicate_final_transcripts"],
        "stale_results": gate["stale_results"],
        "oom": gate["oom"],
        "crashes": gate["crashes"],
        "throttle_reasons": gate["throttle_reasons"],
    }


def _safe_environment() -> tuple[str, Path, Path, Path, str, str]:
    challenge = os.environ.get("CYRUS_PROFILE_QUAL_CHALLENGE", "")
    output = Path(os.environ.get("CYRUS_PROFILE_QUAL_OUTPUT", ""))
    diagnostic = Path(os.environ.get("CYRUS_PROFILE_QUAL_DIAGNOSTIC", ""))
    manifest = Path(os.environ.get("CYRUS_PROFILE_QUAL_MANIFEST", ""))
    base_url = os.environ.get("CYRUS_PROFILE_QUAL_BASE_URL", "")
    api_key = os.environ.get("CYRUS_PROFILE_QUAL_API_KEY", "")
    try:
        parent_info = output.parent.lstat()
    except OSError as exc:
        raise PrimaryRunnerFailure("invalid profile qualification environment") from exc
    parent_reparse = bool(
        os.name == "nt"
        and getattr(parent_info, "st_file_attributes", 0)
        & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    )
    if (
        _CHALLENGE.fullmatch(challenge) is None
        or not output.is_absolute()
        or not diagnostic.is_absolute()
        or not manifest.is_absolute()
        or output == diagnostic
        or output.exists()
        or diagnostic.exists()
        or not stat.S_ISDIR(parent_info.st_mode)
        or output.parent.is_symlink()
        or parent_reparse
        or diagnostic.parent.resolve() != output.parent.resolve()
        or len(api_key.encode("utf-8")) < 32
    ):
        raise PrimaryRunnerFailure("invalid profile qualification environment")
    return challenge, output, diagnostic, manifest, base_url, api_key


def _write_exclusive(path: Path, document: dict[str, Any]) -> None:
    """Atomically publish a complete document without replacing any file."""

    try:
        parent_info = path.parent.lstat()
    except OSError as exc:
        raise PrimaryRunnerFailure("unsafe qualification output path") from exc
    parent_reparse = bool(
        os.name == "nt"
        and getattr(parent_info, "st_file_attributes", 0)
        & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    )
    if (
        not path.is_absolute()
        or path.exists()
        or not stat.S_ISDIR(parent_info.st_mode)
        or path.parent.is_symlink()
        or parent_reparse
    ):
        raise PrimaryRunnerFailure("unsafe qualification output path")
    raw = canonical_bytes(document)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".cyrus-profile-evidence-", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o600)
        os.link(temporary, path)
    except BaseException:
        try:
            temporary.unlink()
        except OSError:
            pass
        raise
    temporary.unlink()


def run() -> dict[str, Any]:
    if os.name != "nt":
        raise PrimaryRunnerFailure("native Windows is required")
    challenge, _output, _diagnostic, manifest_path, base_url, api_key = (
        _safe_environment()
    )
    started_wall = time.time_ns()
    started_mono = time.monotonic_ns()
    manifest = load_manifest(manifest_path)
    if manifest.platform != "windows-x86_64":
        raise PrimaryRunnerFailure("Windows primary manifest is required")
    generic_report = parse_body(manifest.qualification_report.read_bytes())
    if not isinstance(generic_report, dict) or generic_report.get("status") != "pass":
        raise PrimaryRunnerFailure("current generic qualification is required")
    current_identity = generic_report.get("identity")
    if not isinstance(current_identity, dict):
        raise PrimaryRunnerFailure("generic runtime identity is unavailable")
    runtime_identity_sha = hashlib.sha256(
        canonical_bytes(current_identity)
    ).hexdigest()
    quality_path = resolve_local(
        manifest.root, PRIMARY_CORPUS_RELATIVE, "profile_quality_fixture"
    )
    quality_fixtures, quality_digest = load_primary_corpus(quality_path)
    if quality_digest != PRIMARY_CORPUS_SHA256:
        raise PrimaryRunnerFailure("primary quality fixture changed")
    performance_fixture = load_fixture(
        resolve_local(
            manifest.root,
            PERFORMANCE_FIXTURE_RELATIVE,
            "profile_performance_fixture",
        )
    )
    qualification_client = PrimaryQualificationClient(base_url, api_key)
    quality_client = PrimaryQualityClient(
        base_url,
        api_key,
        expected_capabilities={
            "text": True,
            "stream": True,
            "tools": True,
            "json_object": True,
            "json_schema": True,
        },
    )

    _stage("quality")
    quality = run_quality(quality_client, quality_fixtures)
    _stage("performance")
    performance, tokenizer_sha, tokenizations = run_performance(
        qualification_client,
        performance_fixture,
        runtime_identity_sha256=runtime_identity_sha,
        manifest=manifest,
    )
    _stage("soak")
    soak_contract = performance_fixture.document["soak"]
    soak, restart_epochs, cold_loads = run_soak(
        qualification_client,
        performance_fixture.requests,
        tokenizations,
        minimum_seconds=float(soak_contract["minimum_monotonic_seconds"]),
        cycles=soak_contract["minimum_cycles"],
        restart_every=soak_contract["full_restart_every_cycles"],
    )
    _attach_cold_load_observations(performance, cold_loads)

    runner = resolve_local(
        manifest.root, PROFILE_RUNNER_RELATIVE, "profile_fixture_runner"
    )
    _, runner_sha = hash_file(runner)
    identity, _generic_gate = expected_profile_identity(
        manifest,
        current_identity,
        generic_report,
        profile_runner_sha256=runner_sha,
        profile_evidence_sha256="0" * 64,
        tokenizer_identity_sha256=tokenizer_sha,
    )
    identity.pop("profile_evidence_sha256")
    completed_wall = time.time_ns()
    elapsed_ns = time.monotonic_ns() - started_mono
    evidence = {
        "schema": EVIDENCE_SCHEMA,
        "status": "pass",
        "profile_id": "local-primary",
        "platform": "windows-x86_64",
        "attestation_level": ATTESTATION_LEVEL,
        "operator_trust_required": OPERATOR_TRUST_REQUIRED,
        "identity": identity,
        "provenance": {
            "challenge": challenge,
            "started_unix_ns": str(started_wall),
            "completed_unix_ns": str(completed_wall),
            "monotonic_elapsed_ns": elapsed_ns,
            "restart_epochs": restart_epochs,
        },
        "quality": quality,
        "performance": performance,
        "soak": soak,
        "windows_coexistence": _coexistence_from_generic(generic_report),
        "privacy": {
            "runner_content_payloads_persisted": 0,
            "runner_content_payloads_logged": 0,
            "evidence_content_free": True,
            "diagnostic_sanitization_passed": True,
        },
    }
    # Reuse the authoritative validator before publishing. It re-hashes every
    # fixed fixture/runner/generic input and recomputes all aggregate metrics.
    from .profile_report import validate_profile_document

    return validate_profile_document(
        evidence,
        manifest,
        current_identity,
        generic_report,
        evidence=True,
    )


def main() -> int:
    try:
        _challenge, output, _diagnostic, _manifest, _base_url, _api_key = (
            _safe_environment()
        )
        evidence = run()
        _stage("evidence_write")
        _write_exclusive(output, evidence)
        return 0
    except BaseException as exc:
        diagnostic = Path(os.environ.get("CYRUS_PROFILE_QUAL_DIAGNOSTIC", ""))
        if diagnostic.is_absolute() and not diagnostic.exists():
            try:
                _write_exclusive(
                    diagnostic,
                    {
                        "stage": _current_stage,
                        "kind": (
                            "fixture_failure"
                            if isinstance(
                                exc,
                                (PrimaryRunnerFailure, PreflightFailure, ValueError),
                            )
                            else "unexpected"
                        ),
                    },
                )
            except BaseException:
                pass
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
