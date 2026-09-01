from __future__ import annotations

import hashlib
import math
import os
import secrets
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

from .durable import ExclusiveFileLock, durable_replace, read_report
from .engine import EngineFailure, EngineProtocolFailure
from .errors import ApplianceError
from .gateway import GatewayServer, effective_configuration, package_digest
from .manifest import (
    Manifest,
    assert_runtime_platform,
    resolve_local,
    runtime_identity,
    verify_artifacts,
)
from .private_temp import (
    PrivateLaunchDirectory,
    private_temp_environment,
    trusted_windows_directory,
)
from .profile_report import (
    ATTESTATION_LEVEL,
    EVIDENCE_SCHEMA,
    OPERATOR_TRUST_REQUIRED,
    PROFILE_EVIDENCE_RELATIVE,
    PROFILE_RUNNER_RELATIVE,
    REPORT_SCHEMA,
    ProfileQualificationFailure,
    expected_profile_identity,
    profile_report_required,
    validate_profile_document,
)
from .qualification import _QualificationProduction
from .strict_json import canonical_bytes, parse_body
from .windows_process import WindowsProcessFailure, run_windows_job

_PROFILE_RUNNER_SYSTEM_ENV_KEYS = (
    "LANG",
    "LC_ALL",
)


class _PrimaryQualificationProduction(_QualificationProduction):
    def qualification_runtime(self) -> dict[str, Any]:
        result = super().qualification_runtime()
        result["gateway_pid"] = os.getpid()
        return result

    def qualification_arm_generation(self) -> dict[str, Any]:
        if not self.admission.acquire(blocking=False):
            raise ApplianceError(
                429, "server_busy", "The qualification appliance is busy."
            )
        try:
            with self.state_lock:
                engine = self.engine
            arm = getattr(engine, "arm_qualification_generation", None)
            if engine is None or not engine.ready() or not callable(arm):
                raise ApplianceError(
                    503,
                    "qualification_metrics_unavailable",
                    "Native qualification metrics are unavailable.",
                )
            try:
                serial = arm()
            except EngineFailure as exc:
                raise ApplianceError(
                    409,
                    "qualification_metric_state_invalid",
                    "The qualification metric state is invalid.",
                ) from exc
            if type(serial) is not int or serial <= 0:
                raise ApplianceError(
                    503,
                    "qualification_metrics_unavailable",
                    "Native qualification metrics are unavailable.",
                )
            return {"generation_serial": serial}
        finally:
            self.admission.release()

    def qualification_bind_request_start(self, started: float) -> None:
        with self.state_lock:
            engine = self.engine
        bind = getattr(engine, "bind_qualification_request_start", None)
        if engine is None or not engine.ready() or not callable(bind):
            raise ApplianceError(
                503,
                "qualification_metrics_unavailable",
                "Native qualification metrics are unavailable.",
            )
        try:
            bind(started)
        except EngineFailure as exc:
            raise ApplianceError(
                409,
                "qualification_metric_state_invalid",
                "The qualification metric state is invalid.",
            ) from exc

    def qualification_generation_metric(
        self, expected_serial: int
    ) -> dict[str, Any]:
        with self.state_lock:
            engine = self.engine
        consume = getattr(
            engine, "consume_qualification_generation_metric", None
        )
        if engine is None or not engine.ready() or not callable(consume):
            raise ApplianceError(
                503,
                "qualification_metrics_unavailable",
                "Native qualification metrics are unavailable.",
            )
        try:
            metric = consume(expected_serial)
        except EngineFailure as exc:
            raise ApplianceError(
                409,
                "qualification_metric_state_invalid",
                "The qualification metric state is invalid.",
            ) from exc
        fields = {
            "generation_serial",
            "prompt_token_count",
            "prompt_token_ids_sha256",
            "accepted_token_count",
            "accepted_token_ids_sha256",
            "request_ttft_seconds",
            "backend_prompt_seconds",
            "backend_decode_seconds",
            "backend_decode_tokens_per_second",
            "request_to_engine_final_seconds",
        }
        if not isinstance(metric, dict) or set(metric) != fields:
            raise ApplianceError(
                503,
                "qualification_metrics_unavailable",
                "Native qualification metrics are unavailable.",
            )
        digests = (
            metric["prompt_token_ids_sha256"],
            metric["accepted_token_ids_sha256"],
        )
        numeric = {
            name: metric[name]
            for name in (
                "request_ttft_seconds",
                "backend_prompt_seconds",
                "backend_decode_seconds",
                "backend_decode_tokens_per_second",
                "request_to_engine_final_seconds",
            )
        }
        if (
            metric["generation_serial"] != expected_serial
            or type(metric["prompt_token_count"]) is not int
            or not 1 <= metric["prompt_token_count"] <= self.manifest.context_window
            or type(metric["accepted_token_count"]) is not int
            or not 1 <= metric["accepted_token_count"] <= 128
            or any(
                not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
                for digest in digests
            )
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for value in numeric.values()
            )
            or float(numeric["request_ttft_seconds"]) <= 0
            or float(numeric["backend_prompt_seconds"]) < 0
            or float(numeric["backend_decode_seconds"]) <= 0
            or float(numeric["backend_decode_tokens_per_second"]) <= 0
            or float(numeric["request_to_engine_final_seconds"])
            < float(numeric["request_ttft_seconds"])
            or float(numeric["request_ttft_seconds"]) + 0.05
            < float(numeric["backend_prompt_seconds"])
            or float(numeric["request_to_engine_final_seconds"]) + 0.05
            < float(numeric["backend_prompt_seconds"])
            + float(numeric["backend_decode_seconds"])
            or not math.isclose(
                float(numeric["backend_decode_tokens_per_second"]),
                metric["accepted_token_count"]
                / float(numeric["backend_decode_seconds"]),
                rel_tol=1e-9,
                abs_tol=1e-9,
            )
        ):
            raise ApplianceError(
                503,
                "qualification_metrics_unavailable",
                "Native qualification metrics are unavailable.",
            )
        return {name: metric[name] for name in sorted(fields)}

    def qualification_tokenize(self, request: Any) -> dict[str, Any]:
        if not self.admission.acquire(blocking=False):
            raise ApplianceError(
                429, "server_busy", "The qualification appliance is busy."
            )
        try:
            with self.state_lock:
                engine = self.engine
            if engine is None or not engine.ready():
                raise ApplianceError(
                    503,
                    "backend_unavailable",
                    "The qualification backend is unavailable.",
                )
            deadline = time.monotonic() + 120.0
            try:
                prepared = engine.prepare(request, deadline)
                token_ids = prepared.token_ids
                if (
                    not isinstance(token_ids, tuple)
                    or not token_ids
                    or any(type(token) is not int or token < 0 for token in token_ids)
                ):
                    raise EngineProtocolFailure("invalid tokenizer result")
                return {
                    "token_count": len(token_ids),
                    "token_ids_sha256": hashlib.sha256(
                        canonical_bytes(list(token_ids))
                    ).hexdigest(),
                }
            except (EngineFailure, OSError) as exc:
                raise ApplianceError(
                    503,
                    "qualification_tokenization_failed",
                    "The qualification tokenizer probe failed.",
                ) from exc
            finally:
                try:
                    engine.cleanup(time.monotonic() + 10.0)
                except (EngineFailure, OSError) as exc:
                    raise ApplianceError(
                        503,
                        "qualification_tokenization_cleanup_failed",
                        "The qualification tokenizer cleanup failed.",
                    ) from exc
        finally:
            self.admission.release()


def _start_primary_qualification_gateway(
    manifest: Manifest, engine: Any, engine_factory: Any
) -> tuple[str, _PrimaryQualificationProduction, GatewayServer, threading.Thread, str]:
    secret = secrets.token_urlsafe(48)
    production = _PrimaryQualificationProduction(
        manifest, secret, engine_factory, engine=engine
    )
    server = GatewayServer(production, ("127.0.0.1", 0))
    thread = threading.Thread(
        target=server.serve_forever,
        kwargs={"poll_interval": 0.02},
        daemon=True,
    )
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_address[1]}/v1"
    return secret, production, server, thread, base_url


def _prerequisites(
    manifest: Manifest,
    current_identity: dict[str, Any],
    generic_report: dict[str, Any],
) -> None:
    # Placeholder values are used only to exercise every independently
    # verifiable generic/fixture/runner prerequisite before model spawn. The
    # real tokenizer and evidence identities are accepted only from the fresh,
    # challenge-bound native evidence after the run.
    expected_profile_identity(
        manifest,
        current_identity,
        generic_report,
        profile_runner_sha256="1" * 64,
        profile_evidence_sha256="2" * 64,
        tokenizer_identity_sha256="3" * 64,
    )


def _run_profile_fixture(
    runner: Path,
    manifest: Manifest,
    challenge: str,
    base_url: str,
    api_key: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    fixed_runner = resolve_local(
        manifest.root, PROFILE_RUNNER_RELATIVE, "profile_fixture_runner"
    )
    try:
        if runner.absolute().resolve(strict=True) != fixed_runner.resolve(strict=True):
            raise ProfileQualificationFailure("profile fixture runner path mismatch")
    except OSError as exc:
        raise ProfileQualificationFailure("profile fixture runner path is unsafe") from exc
    initial_size = fixed_runner.stat().st_size
    initial_sha = hashlib.sha256(fixed_runner.read_bytes()).hexdigest()
    with PrivateLaunchDirectory(
        manifest.root, prefix="cyrus-primary-profile-qualification-"
    ) as launch:
        if launch.path is None:
            raise ApplianceError(
                503,
                "profile_fixture_runner_failed",
                "The private profile fixture directory is unavailable.",
            )
        temporary = launch.path
        output = temporary / "evidence.json"
        diagnostic = temporary / "diagnostic.json"
        environment = {
            key: os.environ[key]
            for key in _PROFILE_RUNNER_SYSTEM_ENV_KEYS
            if key in os.environ
        }
        environment.update(private_temp_environment(temporary))
        if os.name == "nt":
            windows = str(trusted_windows_directory())
            environment.update({"SYSTEMROOT": windows, "WINDIR": windows})
        environment.update(
            {
                "CYRUS_PROFILE_QUAL_CHALLENGE": challenge,
                "CYRUS_PROFILE_QUAL_OUTPUT": str(output),
                "CYRUS_PROFILE_QUAL_DIAGNOSTIC": str(diagnostic),
                "CYRUS_PROFILE_QUAL_MANIFEST": str(manifest.path),
                "CYRUS_PROFILE_QUAL_BASE_URL": base_url,
                "CYRUS_PROFILE_QUAL_API_KEY": api_key,
            }
        )
        not_before = time.time_ns()
        started = time.monotonic_ns()
        try:
            launch.mark_launch_started()
            if os.name == "nt":
                result = run_windows_job(
                    [str(fixed_runner)],
                    environment=environment,
                    cwd=manifest.root,
                    timeout=12 * 60 * 60,
                )
            else:
                result = subprocess.run(
                    [str(fixed_runner)],
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    env=environment,
                    cwd=manifest.root,
                    timeout=12 * 60 * 60,
                    check=False,
                )
            launch.mark_tree_drained()
        except (OSError, subprocess.TimeoutExpired, WindowsProcessFailure) as exc:
            raise ApplianceError(
                503,
                "profile_fixture_runner_failed",
                "The primary profile fixture runner did not complete.",
            ) from exc
        elapsed = time.monotonic_ns() - started
        not_after = time.time_ns()
        if (
            result.returncode != 0
            or not output.is_file()
            or output.is_symlink()
            or output.stat().st_size <= 0
            or output.stat().st_size > 2_097_152
        ):
            raise ApplianceError(
                503,
                "profile_fixture_runner_failed",
                "The primary profile fixture runner did not produce fresh evidence.",
            )
        try:
            evidence = parse_body(output.read_bytes())
        except (OSError, ApplianceError) as exc:
            raise ApplianceError(
                503,
                "profile_fixture_evidence_invalid",
                "The primary profile evidence is invalid.",
            ) from exc
        if (
            fixed_runner.stat().st_size != initial_size
            or hashlib.sha256(fixed_runner.read_bytes()).hexdigest() != initial_sha
        ):
            raise ApplianceError(
                503,
                "profile_fixture_runner_identity_changed",
                "The primary profile fixture runner changed during qualification.",
            )
    return evidence, {
        "challenge": challenge,
        "not_before_unix_ns": not_before,
        "not_after_unix_ns": not_after,
        "max_elapsed_ns": elapsed,
    }


def qualify_primary_profile(
    manifest: Manifest,
    engine_factory: Any,
    fixture_runner: Path,
) -> dict[str, Any]:
    if not profile_report_required(manifest) or manifest.profile_qualification_report is None:
        raise ApplianceError(
            503,
            "profile_qualification_not_applicable",
            "The manifest is not the exact Windows local-primary profile.",
        )
    report_path = manifest.profile_qualification_report
    lock_path = report_path.with_suffix(report_path.suffix + ".lock")
    with ExclusiveFileLock(lock_path):
        identity = runtime_identity(
            manifest, package_digest(), effective_configuration()
        )
        generic_report = read_report(manifest.qualification_report)
        if not isinstance(generic_report, dict):
            raise ApplianceError(
                503,
                "generic_qualification_required",
                "A current generic qualification pass is required first.",
            )
        started = int(time.time())
        challenge = secrets.token_hex(32)
        durable_replace(
            report_path,
            {
                "schema": REPORT_SCHEMA,
                "status": "in_progress",
                "profile_id": "local-primary",
                "platform": "windows-x86_64",
                "attestation_level": ATTESTATION_LEVEL,
                "operator_trust_required": OPERATOR_TRUST_REQUIRED,
                "authoritative": False,
                "production_ready": False,
                "started": started,
            },
        )
        engine = None
        try:
            _prerequisites(manifest, identity, generic_report)
            assert_runtime_platform(manifest)
            verify_artifacts(manifest)
            engine = engine_factory(manifest)
            start = getattr(engine, "start", None)
            if callable(start):
                start(time.monotonic() + 600.0)
            if not engine.ready():
                raise ApplianceError(
                    503,
                    "backend_unavailable",
                    "The managed engine did not become ready for profile qualification.",
                )
            secret, production, server, thread, base_url = (
                _start_primary_qualification_gateway(
                    manifest, engine, engine_factory
                )
            )
            try:
                evidence, expected_provenance = _run_profile_fixture(
                    fixture_runner, manifest, challenge, base_url, secret
                )
            finally:
                try:
                    server.shutdown()
                    server.server_close()
                    thread.join(timeout=10)
                finally:
                    production.close()
                    production.require_teardown_proof()
            validated_evidence = validate_profile_document(
                evidence,
                manifest,
                identity,
                generic_report,
                evidence=True,
                expected_provenance=expected_provenance,
            )
            evidence_path = resolve_local(
                manifest.root,
                PROFILE_EVIDENCE_RELATIVE,
                "profile_qualification_evidence",
                must_exist=False,
            )
            durable_replace(evidence_path, validated_evidence)
            evidence_sha = hashlib.sha256(evidence_path.read_bytes()).hexdigest()
            final_identity = dict(validated_evidence["identity"])
            final_identity["profile_evidence_sha256"] = evidence_sha
            report = {
                **validated_evidence,
                "schema": REPORT_SCHEMA,
                "authoritative": True,
                "production_ready": True,
                "identity": final_identity,
            }
            validated_report = validate_profile_document(
                report,
                manifest,
                identity,
                generic_report,
                evidence=False,
                profile_evidence_sha256=evidence_sha,
            )
            durable_replace(report_path, validated_report)
            return validated_report
        except BaseException as exc:
            code = (
                exc.code
                if isinstance(exc, ApplianceError)
                else "profile_qualification_failed"
            )
            durable_replace(
                report_path,
                {
                    "schema": REPORT_SCHEMA,
                    "status": "failed",
                    "profile_id": "local-primary",
                    "platform": "windows-x86_64",
                    "attestation_level": ATTESTATION_LEVEL,
                    "operator_trust_required": OPERATOR_TRUST_REQUIRED,
                    "authoritative": False,
                    "production_ready": False,
                    "started": started,
                    "completed": int(time.time()),
                    "failure_code": code,
                    "runtime_identity_sha256": hashlib.sha256(
                        canonical_bytes(identity)
                    ).hexdigest(),
                },
            )
            raise
        finally:
            if engine is not None:
                engine.close()
