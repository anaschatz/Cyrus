from __future__ import annotations

import hashlib
import asyncio
import math
import os
import re
import secrets
import subprocess
import tempfile
from decimal import Decimal
import threading
import time
from pathlib import Path
from typing import Any

from .durable import ExclusiveFileLock, durable_replace
from .errors import ApplianceError, invalid
from .gateway import effective_configuration, package_digest
from .gateway import GatewayServer, Production
from .manifest import Manifest, assert_runtime_platform, resolve_local, runtime_identity, verify_artifacts
from .strict_json import parse_body
from .strict_json import canonical_bytes
from .limits import BODY_BYTES

SUITE_VERSION = "jarvis-inference-qualification-v1"


def _finite_number(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float, Decimal)) and math.isfinite(float(value))


def _exact(value: Any, fields: set[str], name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise invalid("invalid_qualification_evidence", "Qualification evidence has an invalid exact shape.", name)
    return value


def load_evidence(path: Path) -> tuple[dict[str, Any], str]:
    if path.stat().st_size > BODY_BYTES:
        raise invalid("invalid_qualification_evidence", "Qualification evidence exceeds its size limit.")
    raw = path.read_bytes()
    value = parse_body(raw)
    return value, hashlib.sha256(raw).hexdigest()


def evaluate_evidence(
    manifest: Manifest,
    evidence: dict[str, Any],
    identity: dict[str, Any],
    expected_provenance: dict[str, Any],
) -> dict[str, Any]:
    required = {
        "suite_version", "fixture_digest", "deterministic_seeds", "identity", "provenance",
        "streaming", "tools", "json_schema", "cancellation", "crash_restart",
        "isolation", "protocol_conformance", "privacy_leaks", "unexpected_oom",
        "unexpected_crashes", "unexpected_hangs", "platform_gate",
    }
    item = _exact(evidence, required, "evidence")
    if item["suite_version"] != SUITE_VERSION or item["identity"] != identity:
        raise invalid("qualification_identity_mismatch", "Qualification evidence does not match this runtime identity.")
    if not isinstance(item["fixture_digest"], str) or len(item["fixture_digest"]) != 64 or any(c not in "0123456789abcdef" for c in item["fixture_digest"]):
        raise invalid("invalid_qualification_evidence", "The fixture digest is invalid.")
    provenance = _exact(item["provenance"], {
        "challenge", "identity_digest", "fixture_digest", "started_unix_ns",
        "completed_unix_ns", "monotonic_elapsed_ns",
    }, "provenance")
    if provenance["challenge"] != expected_provenance["challenge"] or provenance["identity_digest"] != expected_provenance["identity_digest"] or provenance["fixture_digest"] != item["fixture_digest"]:
        raise invalid("qualification_provenance_mismatch", "Qualification evidence is not bound to this exclusive run.")
    for key in ("started_unix_ns", "completed_unix_ns"):
        if not isinstance(provenance[key], str) or not re.fullmatch(r"[1-9][0-9]{0,19}", provenance[key]):
            raise invalid("invalid_qualification_evidence", "Qualification provenance timestamps are invalid.")
    if type(provenance["monotonic_elapsed_ns"]) is not int or provenance["monotonic_elapsed_ns"] <= 0:
        raise invalid("invalid_qualification_evidence", "Qualification provenance timestamps are invalid.")
    started_unix_ns = int(provenance["started_unix_ns"])
    completed_unix_ns = int(provenance["completed_unix_ns"])
    if started_unix_ns < expected_provenance["not_before_unix_ns"] or completed_unix_ns > expected_provenance["not_after_unix_ns"] or completed_unix_ns < started_unix_ns:
        raise invalid("qualification_provenance_mismatch", "Qualification evidence falls outside this run boundary.")
    if provenance["monotonic_elapsed_ns"] > expected_provenance["max_elapsed_ns"]:
        raise invalid("qualification_provenance_mismatch", "Qualification monotonic provenance is invalid.")
    if not isinstance(item["deterministic_seeds"], list) or not item["deterministic_seeds"] or any(type(seed) is not int for seed in item["deterministic_seeds"]):
        raise invalid("invalid_qualification_evidence", "Deterministic seeds are required.")
    streaming = _exact(item["streaming"], {"greek_success", "english_success"}, "streaming")
    if streaming != {"greek_success": 20, "english_success": 20}:
        raise invalid("qualification_gate_failed", "The 20 Greek and 20 English streaming gates did not pass.")
    tools = _exact(item["tools"], {"named_success", "required_success"}, "tools")
    expected_tool = {"named_success": 10, "required_success": 10} if manifest.capabilities.tools else {"named_success": 0, "required_success": 0}
    if tools != expected_tool:
        raise invalid("qualification_gate_failed", "The capability-qualified tool gate did not pass.")
    schemas = _exact(item["json_schema"], {"strict_success"}, "json_schema")
    expected_schema = 20 if manifest.capabilities.json_schema else 0
    if schemas != {"strict_success": expected_schema}:
        raise invalid("qualification_gate_failed", "The capability-qualified JSON Schema gate did not pass.")
    cancellation = _exact(item["cancellation"], {"prefill_success", "mid_stream_success"}, "cancellation")
    if cancellation != {"prefill_success": 5, "mid_stream_success": 5}:
        raise invalid("qualification_gate_failed", "The cancellation gate did not pass.")
    crashes = _exact(item["crash_restart"], {"isolated_success"}, "crash_restart")
    isolation = _exact(item["isolation"], {"pairs_success", "leaks"}, "isolation")
    if crashes != {"isolated_success": 5} or isolation != {"pairs_success": 50, "leaks": 0}:
        raise invalid("qualification_gate_failed", "The restart or semantic-isolation gate did not pass.")
    protocol = item["protocol_conformance"]
    if not _finite_number(protocol) or protocol != 1 or any(type(item[key]) is not int or item[key] != 0 for key in ("privacy_leaks", "unexpected_oom", "unexpected_crashes", "unexpected_hangs")):
        raise invalid("qualification_gate_failed", "Protocol, privacy, or stability qualification failed.")
    _platform_gate(manifest, item["platform_gate"], identity)
    return {
        "suite_version": item["suite_version"],
        "fixture_digest": item["fixture_digest"],
        "deterministic_seeds": item["deterministic_seeds"],
        "provenance": provenance,
        "results": {key: item[key] for key in required - {"suite_version", "fixture_digest", "deterministic_seeds", "identity", "provenance"}},
    }


def _platform_gate(manifest: Manifest, value: Any, identity: dict[str, Any]) -> None:
    if manifest.platform == "windows-x86_64":
        gate = _exact(value, {"kind", "fixture", "audio_corpus_digest", "greek_utterances", "english_utterances", "baseline_power_identity_digest", "coexistence_power_identity_digest", "overlapping_utterances", "baseline_p95", "coexistence_p95", "lost_final_transcripts", "oom", "crashes", "throttle_reasons"}, "platform_gate")
        if gate["kind"] != "windows_asr_coexistence":
            raise invalid("qualification_gate_failed", "The Windows ASR coexistence gate is missing.")
        fixture = _exact(gate["fixture"], {"rendered_prompt_token_lengths", "max_completion_tokens", "request_count", "concurrency", "schedule", "minimum_overlapping_utterances", "sampling", "power_profile"}, "platform_gate.fixture")
        power = _exact(fixture["power_profile"], {"windows_power_scheme_guid", "ac_state", "gpu_power_limit_watts", "gpu_clock_policy", "driver_version", "compute_runtime", "require_zero_thermal_power_throttle_reasons"}, "platform_gate.fixture.power_profile")
        guid = power["windows_power_scheme_guid"]
        if not isinstance(guid, str) or not re.fullmatch(r"[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}", guid) or power["ac_state"] != "ac" or not _finite_number(power["gpu_power_limit_watts"]) or power["gpu_power_limit_watts"] <= 0 or any(not isinstance(power[key], str) or not power[key] for key in ("gpu_clock_policy", "driver_version", "compute_runtime")) or power["require_zero_thermal_power_throttle_reasons"] is not True:
            raise invalid("qualification_gate_failed", "The Windows power identity is invalid.")
        hardware = identity.get("hardware_runtime", {})
        if power["driver_version"] != hardware.get("driver") or power["compute_runtime"] != hardware.get("compute_runtime"):
            raise invalid("qualification_gate_failed", "The Windows power fixture runtime identity does not match qualification.")
        if fixture["rendered_prompt_token_lengths"] != [512, 2048, 4096] or fixture["max_completion_tokens"] != 512 or fixture["request_count"] != 100 or fixture["concurrency"] != 1 or fixture["schedule"] != "serial-continuous-from-audio-start" or fixture["minimum_overlapping_utterances"] != 80 or fixture["sampling"] != {"temperature": 0, "top_p": 1}:
            raise invalid("qualification_gate_failed", "The hashed Windows ASR fixture is invalid.")
        power_digest = hashlib.sha256(canonical_bytes(power)).hexdigest()
        if not isinstance(gate["audio_corpus_digest"], str) or not re.fullmatch(r"[0-9a-f]{64}", gate["audio_corpus_digest"]) or gate["greek_utterances"] != 50 or gate["english_utterances"] != 50 or gate["baseline_power_identity_digest"] != power_digest or gate["coexistence_power_identity_digest"] != power_digest:
            raise invalid("qualification_gate_failed", "The Windows corpus or baseline/coexistence power identity is invalid.")
        baseline = gate["baseline_p95"]
        coexistence = gate["coexistence_p95"]
        if not _finite_number(baseline) or baseline <= 0 or not _finite_number(coexistence) or coexistence <= 0 or coexistence / baseline - 1 > Decimal("0.15"):
            raise invalid("qualification_gate_failed", "ASR coexistence degradation exceeds 15 percent.")
        if any(type(gate[key]) is not int for key in ("overlapping_utterances", "lost_final_transcripts", "oom", "crashes", "throttle_reasons")) or gate["overlapping_utterances"] < 80 or gate["lost_final_transcripts"] != 0 or gate["oom"] != 0 or gate["crashes"] != 0 or gate["throttle_reasons"] != 0:
            raise invalid("qualification_gate_failed", "The Windows ASR stability gate failed.")
    else:
        gate = _exact(value, {"kind", "warmup_requests", "normal_requests", "final_drift", "theil_sen_projected_growth", "oom", "crashes", "stale_state_references"}, "platform_gate")
        if gate["kind"] != "mac_unified_memory" or gate["warmup_requests"] != 20 or gate["normal_requests"] != 200:
            raise invalid("qualification_gate_failed", "The Mac unified-memory run is incomplete.")
        if not _finite_number(gate["final_drift"]) or gate["final_drift"] > Decimal("0.05") or not _finite_number(gate["theil_sen_projected_growth"]) or gate["theil_sen_projected_growth"] > Decimal("0.05"):
            raise invalid("qualification_gate_failed", "The Mac unified-memory drift gate failed.")
        if gate["oom"] != 0 or gate["crashes"] != 0 or gate["stale_state_references"] != 0:
            raise invalid("qualification_gate_failed", "The Mac unified-memory stability gate failed.")


def qualify(manifest: Manifest, engine_factory: Any, fixture_runner: Path) -> dict[str, Any]:
    lock_path = manifest.qualification_report.with_suffix(manifest.qualification_report.suffix + ".lock")
    with ExclusiveFileLock(lock_path):
        identity = runtime_identity(manifest, package_digest(), effective_configuration())
        started = int(time.time())
        challenge = secrets.token_hex(32)
        identity_digest = hashlib.sha256(canonical_identity(identity)).hexdigest()
        durable_replace(manifest.qualification_report, {
            "report_schema_version": 1, "status": "in_progress", "suite_version": SUITE_VERSION,
            "started": started, "identity": identity, "challenge": challenge,
        })
        engine = None
        try:
            assert_runtime_platform(manifest)
            gateway_runtime = identity.get("gateway_runtime", {})
            wheel = gateway_runtime.get("wheel") if isinstance(gateway_runtime, dict) else None
            if (
                not isinstance(gateway_runtime, dict)
                or gateway_runtime.get("version") != "1.0.0"
                or gateway_runtime.get("openai_sdk_version") != "2.46.0"
                or not isinstance(wheel, dict)
                or not isinstance(wheel.get("sha256"), str)
                or not re.fullmatch(r"[0-9a-f]{64}", wheel["sha256"])
            ):
                raise ApplianceError(503, "runtime_identity_incomplete", "The pinned gateway wheel and OpenAI SDK are required for native qualification.")
            hardware = identity["hardware_runtime"]
            if (
                not identity.get("cpu") or not isinstance(hardware.get("ram_bytes"), int) or hardware["ram_bytes"] <= 0
                or any(not isinstance(hardware.get(key), str) or not hardware[key] or hardware[key] == "unknown" for key in ("gpu_device", "driver", "compute_runtime", "compute_capability"))
                or not isinstance(hardware.get("gpu_memory_bytes"), int) or hardware["gpu_memory_bytes"] <= 0
            ):
                raise ApplianceError(503, "runtime_identity_incomplete", "Native GPU runtime identity is incomplete.")
            verify_artifacts(manifest)
            engine = engine_factory(manifest)
            start = getattr(engine, "start", None)
            if callable(start):
                start(time.monotonic() + 600.0)
            if not engine.ready():
                raise ApplianceError(503, "backend_unavailable", "The managed engine did not become ready.")
            secret, production, server, thread, base_url = _start_qualification_gateway(manifest, engine, engine_factory)
            try:
                _openai_sdk_bootstrap(manifest, base_url, secret)
                evidence, evidence_digest, runner_identity, expected_provenance = _run_fixture_runner(
                    fixture_runner, manifest, challenge, identity_digest, base_url, secret
                )
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=10)
                production.close()
            summary = evaluate_evidence(manifest, evidence, identity, expected_provenance)
            report = {
                "report_schema_version": 1, "status": "pass", "started": started, "completed": int(time.time()),
                "identity": identity, "evidence_digest": evidence_digest,
                "fixture_runner": runner_identity, **summary,
            }
            durable_replace(manifest.qualification_report, report)
            return report
        except BaseException as exc:
            code = exc.code if isinstance(exc, ApplianceError) else "qualification_failed"
            failed = {
                "report_schema_version": 1, "status": "failed", "started": started, "completed": int(time.time()),
                "identity": identity, "failure_code": code,
            }
            durable_replace(manifest.qualification_report, failed)
            raise
        finally:
            if engine is not None:
                engine.close()


def canonical_identity(identity: dict[str, Any]) -> bytes:
    return canonical_bytes(identity)


def _run_fixture_runner(
    runner: Path,
    manifest: Manifest,
    challenge: str,
    identity_digest: str,
    base_url: str,
    api_key: str,
) -> tuple[dict[str, Any], str, dict[str, Any], dict[str, Any]]:
    supplied = runner.absolute()
    try:
        relative = supplied.relative_to(manifest.root).as_posix()
    except ValueError as exc:
        raise ApplianceError(503, "fixture_runner_untrusted", "The native fixture runner must be local to the manifest root.") from exc
    try:
        runner = resolve_local(manifest.root, relative, "fixture_runner")
    except ApplianceError as exc:
        raise ApplianceError(503, "fixture_runner_untrusted", "The native fixture runner path is unsafe.") from exc
    runner_size = runner.stat().st_size
    runner_digest = hashlib.sha256(runner.read_bytes()).hexdigest()
    with tempfile.TemporaryDirectory(prefix="jarvis-qualification-", dir=manifest.root) as temporary:
        output = Path(temporary) / "evidence.json"
        not_before_wall = time.time_ns()
        started_mono = time.monotonic_ns()
        environment = {
            key: os.environ[key]
            for key in ("PATH", "SYSTEMROOT", "WINDIR", "TEMP", "TMP", "TMPDIR", "LANG", "LC_ALL")
            if key in os.environ
        }
        environment.update({
            "JARVIS_QUAL_CHALLENGE": challenge,
            "JARVIS_QUAL_IDENTITY_DIGEST": identity_digest,
            "JARVIS_QUAL_OUTPUT": str(output),
            "JARVIS_QUAL_MANIFEST": str(manifest.path),
            "JARVIS_QUAL_BASE_URL": base_url,
            "JARVIS_QUAL_API_KEY": api_key,
        })
        if manifest.platform == "macos-arm64":
            executable = resolve_local(
                manifest.root,
                manifest.python["executable"]["path"],
                "python.executable",
            )
            command = [str(executable), "-I", str(runner)]
        else:
            command = [str(runner)]
        try:
            result = subprocess.run(
                command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL, env=environment, cwd=manifest.root,
                timeout=24 * 60 * 60, check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise ApplianceError(503, "fixture_runner_failed", "The native fixture runner did not complete successfully.") from exc
        not_after_wall = time.time_ns()
        elapsed = time.monotonic_ns() - started_mono
        if result.returncode != 0 or not output.is_file() or output.is_symlink():
            raise ApplianceError(503, "fixture_runner_failed", "The native fixture runner did not produce fresh evidence.")
        evidence, evidence_digest = load_evidence(output)
        if runner.stat().st_size != runner_size or hashlib.sha256(runner.read_bytes()).hexdigest() != runner_digest:
            raise ApplianceError(503, "fixture_runner_identity_changed", "The native fixture runner changed during qualification.")
    expected = {
        "challenge": challenge,
        "identity_digest": identity_digest,
        "not_before_unix_ns": not_before_wall,
        "not_after_unix_ns": not_after_wall,
        "max_elapsed_ns": elapsed,
    }
    runner_identity = {"path": relative, "size_bytes": runner_size, "sha256": runner_digest}
    return evidence, evidence_digest, runner_identity, expected


class _QualificationProduction(Production):
    def __post_init__(self) -> None:
        super().__post_init__()
        self._qualification_history: list[dict[str, Any]] = []
        self._qualification_seen: set[tuple[int, int]] = set()

    def start_background(self) -> None:
        with self.state_lock:
            if self.loader is not None:
                return
            process = getattr(self.engine, "process", None)
            if process is not None:
                setattr(self.engine, "_qualification_original_pid", process.pid)
            self.ready_state = self.engine is not None and self.engine.ready()
            self.loader = threading.Thread(target=self._qualification_supervise, name="jarvis-qualification-loader", daemon=True)
            self.loader.start()

    def _record_termination(self, engine: Any) -> None:
        snapshot = getattr(engine, "runtime_snapshot", lambda: {})()
        serial = snapshot.get("termination_serial")
        pid = snapshot.get("pid")
        if type(serial) is not int or serial <= 0:
            return
        original_pid = getattr(engine, "_qualification_original_pid", None)
        identity = (int(original_pid or pid or 0), serial)
        if identity in self._qualification_seen:
            return
        self._qualification_seen.add(identity)
        self._qualification_history.append({
            "pid": identity[0],
            "kind": snapshot.get("last_termination_kind"),
            "phase": snapshot.get("last_termination_phase"),
            "elapsed_seconds": snapshot.get("last_termination_elapsed"),
            "exit_ok": snapshot.get("last_termination_exit_ok"),
        })
        self._qualification_history[:] = self._qualification_history[-64:]

    def _qualification_supervise(self) -> None:
        while not self.stop.is_set():
            with self.state_lock:
                active = self.engine
            if active is None or not active.ready():
                if active is not None:
                    self._record_termination(active)
                    active.close()
                candidate = None
                try:
                    candidate = self.engine_factory(self.manifest)
                    with self.state_lock:
                        self.engine = None
                        self.loading_engine = candidate
                        self.ready_state = False
                    start = getattr(candidate, "start", None)
                    if callable(start):
                        start(time.monotonic() + 600.0)
                    process = getattr(candidate, "process", None)
                    if process is not None:
                        setattr(candidate, "_qualification_original_pid", process.pid)
                    with self.state_lock:
                        self.engine = candidate
                        self.loading_engine = None
                        self.ready_state = candidate.ready()
                except Exception:
                    if candidate is not None:
                        candidate.close()
                    with self.state_lock:
                        self.loading_engine = None
                        self.ready_state = False
            self.stop.wait(0.05)

    def is_ready(self) -> bool:
        return self.engine is not None and self.engine.ready()

    def refresh(self) -> bool:
        return self.is_ready()

    def qualification_runtime(self) -> dict[str, Any]:
        with self.state_lock:
            engine = self.engine
            ready = bool(self.ready_state and engine is not None and engine.ready())
        snapshot = getattr(engine, "runtime_snapshot", lambda: {})() if engine is not None else {}
        return {
            "ready": ready,
            "busy": self.admission.locked(),
            "engine": snapshot,
            "terminations": list(self._qualification_history),
        }

    def inject_qualification_crash(self) -> dict[str, Any]:
        with self.state_lock:
            engine = self.engine
        inject = getattr(engine, "inject_crash", None)
        if engine is None or not engine.ready() or not callable(inject):
            raise ApplianceError(503, "backend_unavailable", "The qualification backend is unavailable.")
        process = getattr(engine, "process", None)
        if process is not None:
            setattr(engine, "_qualification_original_pid", process.pid)
        result = inject()
        self._record_termination(engine)
        return result

    def inject_qualification_cancel(self, phase: str) -> dict[str, Any]:
        with self.state_lock:
            engine = self.engine
        snapshot = getattr(engine, "runtime_snapshot", lambda: {})() if engine is not None else {}
        if engine is None or not engine.ready() or snapshot.get("phase") != phase:
            raise ApplianceError(409, "qualification_phase_mismatch", "The qualification backend is not in the requested phase.")
        process = getattr(engine, "process", None)
        pid = process.pid if process is not None else None
        if process is not None:
            setattr(engine, "_qualification_original_pid", process.pid)
        engine.cancel()
        self._record_termination(engine)
        result = getattr(engine, "runtime_snapshot", lambda: {})()
        return {
            "pid": pid,
            "phase": result.get("last_termination_phase"),
            "elapsed_seconds": result.get("last_termination_elapsed"),
            "exit_ok": result.get("last_termination_exit_ok"),
        }


def _start_qualification_gateway(manifest: Manifest, engine: Any, engine_factory: Any):
    secret = secrets.token_urlsafe(48)
    production = _QualificationProduction(manifest, secret, engine_factory, engine=engine)
    server = GatewayServer(production, ("127.0.0.1", 0))
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_address[1]}/v1"
    return secret, production, server, thread, base_url


def _openai_sdk_bootstrap(manifest: Manifest, base_url: str, secret: str) -> None:
    """Exercise a real ephemeral gateway with both official SDK clients.

    This is deliberately a bootstrap check, not a replacement for the complete
    hashed native fixture suite summarized by the evidence file.
    """
    try:
        from openai import AsyncOpenAI, OpenAI
    except ImportError as exc:
        raise ApplianceError(503, "qualification_dependency_missing", "The pinned OpenAI Python SDK is required for qualification.") from exc
    request = {
        "model": manifest.alias,
        "messages": [{"role": "user", "content": "Reply with one short word."}],
        "max_completion_tokens": 8,
    }
    client = OpenAI(base_url=base_url, api_key=secret, timeout=300.0, max_retries=0)
    try:
        completion = client.chat.completions.create(**request)
        if completion.model != manifest.alias or len(completion.choices) != 1 or completion.choices[0].finish_reason not in {"stop", "length", "tool_calls"} or completion.usage is None:
            raise ApplianceError(503, "sdk_conformance_failed", "The synchronous SDK bootstrap failed.")
        stream = client.chat.completions.create(**request, stream=True, stream_options={"include_usage": True})
        chunks = list(stream)
        if not chunks or chunks[-1].usage is None:
            raise ApplianceError(503, "sdk_conformance_failed", "The streaming SDK bootstrap failed.")

        async def check_async() -> None:
            async_client = AsyncOpenAI(base_url=base_url, api_key=secret, timeout=300.0, max_retries=0)
            try:
                result = await async_client.chat.completions.create(**request)
                if result.model != manifest.alias or result.usage is None:
                    raise ApplianceError(503, "sdk_conformance_failed", "The asynchronous SDK bootstrap failed.")
                async_stream = await async_client.chat.completions.create(
                    **request, stream=True, stream_options={"include_usage": True}
                )
                chunks = [chunk async for chunk in async_stream]
                if not chunks or chunks[-1].usage is None:
                    raise ApplianceError(503, "sdk_conformance_failed", "The asynchronous streaming SDK bootstrap failed.")
            finally:
                await async_client.close()

        asyncio.run(check_async())
    finally:
        client.close()
