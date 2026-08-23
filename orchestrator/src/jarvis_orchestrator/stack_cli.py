"""One content-free operator entry point for the private local CYRUS stack."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from .config import OrchestratorConfig
from .inference import (
    InferenceControlClient,
    InferenceIdentityEndpointUnavailable,
)
from .macos_beta_lease import validate_live_session_lease
from .readiness import (
    run_live_model_failure_gate,
    run_live_sequential_gate,
    write_readiness_report,
)
from .service import (
    BETA_SESSION_SCHEMA_VERSION,
    valid_voice_connection_tracker_id,
)
from .stack_config import (
    STACK_SCHEMA_VERSION,
    PrivateStackConfig,
    StackConfigError,
    write_private_stack_config,
)


def _voice_auth_key(config: OrchestratorConfig):
    from jarvis_voice.voice_auth import VoiceAuthError, load_voice_auth_key

    if config.voice_auth_key_file is None:
        raise StackConfigError("voice_auth_key_required")
    try:
        return load_voice_auth_key(config.voice_auth_key_file)
    except VoiceAuthError:
        raise StackConfigError("voice_auth_key_invalid") from None


def _print(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cyrus-stack")
    parser.add_argument("--config", type=Path)
    commands = parser.add_subparsers(dest="command", required=True)
    initialize = commands.add_parser("init")
    initialize.add_argument("--output", type=Path, required=True)
    initialize.add_argument("--voice-config", type=Path, required=True)
    initialize.add_argument("--orchestrator-config", type=Path, required=True)
    initialize.add_argument("--inference-runner", type=Path, required=True)
    initialize.add_argument("--inference-manifest", type=Path, required=True)
    initialize.add_argument(
        "--inference-qualification-report", type=Path, required=True
    )
    migrate = commands.add_parser("prepare-auth-v2")
    migrate.add_argument("--legacy-config", type=Path, required=True)
    migrate.add_argument("--output-directory", type=Path, required=True)
    migrate.add_argument(
        "--macos-ptt-key",
        choices=("CONTROL_OPTION", "RIGHT_OPTION"),
    )
    migrate.add_argument("--safe-beta-scope", action="store_true")
    migrate.add_argument(
        "--rotate-weak-bearer-tokens",
        action="store_true",
        help=(
            "replace non-canonical legacy inline bearer tokens only in the "
            "fresh v2 output bundle"
        ),
    )
    commands.add_parser("validate")
    commands.add_parser("status")
    preflight = commands.add_parser("beta-preflight")
    preflight.add_argument("--output", type=Path)
    beta_run = commands.add_parser("beta-run")
    beta_run.add_argument(
        "--runtime-python", type=Path, default=Path(sys.executable)
    )
    beta_run.add_argument("--session-parent", type=Path, required=True)
    beta_run.add_argument(
        "--startup-timeout-seconds", type=float, default=600.0
    )
    beta_recover = commands.add_parser("beta-recover")
    beta_recover.add_argument("--session-parent", type=Path, required=True)
    physical_turn = commands.add_parser(
        "physical-turn",
        help="verify one operator-observed turn over the authenticated Voice transport",
        description=(
            "Arm a content-free gate while the operator performs exactly one macOS "
            "PTT turn. This verifies authenticated transport evidence, not native "
            "binary attestation."
        ),
    )
    physical_turn.add_argument("--timeout-seconds", type=float, default=180.0)
    physical_turn.add_argument("--poll-interval-seconds", type=float, default=0.1)
    physical_turn.add_argument("--output", type=Path)
    e2e = commands.add_parser("e2e")
    e2e.add_argument("--turns", type=int, default=1)
    e2e.add_argument("--output", type=Path)
    failure = commands.add_parser("model-failure")
    failure.add_argument("--timeout-seconds", type=float, required=True)
    failure.add_argument("--output", type=Path)
    return parser


def _load(parsed: argparse.Namespace) -> PrivateStackConfig:
    if parsed.config is None:
        raise StackConfigError("stack_config_required")
    return PrivateStackConfig.load(parsed.config)


def _beta_session_error_payload(
    code: str,
    *,
    preflight_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a content-free CLI error while preserving native probe causes."""

    error_code = code
    cause_codes: list[str] = []
    if code == "beta_preflight_failed" and isinstance(preflight_report, dict):
        raw = preflight_report.get("voice_environment_blockers")
        if (
            isinstance(raw, list)
            and len(raw) <= 32
            and all(
                isinstance(item, str)
                and 1 <= len(item) <= 128
                and item[0].isalpha()
                and all(
                    character
                    in "abcdefghijklmnopqrstuvwxyz0123456789_.-"
                    for character in item
                )
                for item in raw
            )
        ):
            cause_codes = sorted(set(raw))
        if "environment.restricted_native_io_probe" in cause_codes:
            error_code = "beta_preflight_native_probe_inconclusive"
    error: dict[str, Any] = {"code": error_code, "retryable": False}
    if cause_codes:
        error["cause_codes"] = cause_codes
    return {"error": error, "ready": False}


def _readyz(config: OrchestratorConfig) -> bool:
    request = urllib.request.Request(
        f"http://{config.control_host}:{config.control_port}/readyz",
        headers={"Host": f"127.0.0.1:{config.control_port}"},
    )
    try:
        with urllib.request.urlopen(request, timeout=2) as response:
            payload = json.loads(response.read())
        return response.status == 200 and payload == {"ready": True}
    except (OSError, ValueError, urllib.error.URLError):
        return False


def _beta_session(config: OrchestratorConfig) -> dict[str, Any] | None:
    request = urllib.request.Request(
        f"http://{config.control_host}:{config.control_port}/v1/beta-session",
        headers={
            "Accept": "application/json",
            "Authorization": f"Bearer {config.control_bearer_token}",
            "Host": f"127.0.0.1:{config.control_port}",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=2) as response:
            raw = response.read(16 * 1024 + 1)
        if response.status != 200 or len(raw) > 16 * 1024:
            return None
        payload = json.loads(raw.decode("utf-8", errors="strict"))
        expected = {
            "core_ready",
            "effective_capabilities",
            "schema_version",
            "voice_client_connected",
            "voice_connection_generation",
            "voice_connection_tracker_id",
        }
        capability_names = {
            "cloud",
            "memory_retrospective",
            "memory_shadow",
            "research",
            "tools",
        }
        if not isinstance(payload, dict) or set(payload) != expected:
            return None
        capabilities = payload.get("effective_capabilities")
        if (
            payload.get("schema_version")
            != BETA_SESSION_SCHEMA_VERSION
            or type(payload.get("core_ready")) is not bool
            or type(payload.get("voice_client_connected")) is not bool
            or type(payload.get("voice_connection_generation")) is not int
            or payload["voice_connection_generation"] < 0
            or not valid_voice_connection_tracker_id(
                payload.get("voice_connection_tracker_id")
            )
            or not isinstance(capabilities, dict)
            or set(capabilities) != capability_names
            or any(type(capabilities[name]) is not bool for name in capability_names)
        ):
            return None
        return payload
    except (OSError, UnicodeError, ValueError, urllib.error.URLError):
        return None


async def _voice_auth_ready(config: OrchestratorConfig) -> bool:
    try:
        from websockets.asyncio.client import connect
        from jarvis_voice.voice_auth_transport import (
            AUTH_PROBE_FRAME,
            AUTH_PROBE_PATH,
            AUTH_PROBE_SUCCESS_FRAME,
            authenticate_async_client,
        )

        uri = f"ws://{config.voice_host}:{config.voice_port}{AUTH_PROBE_PATH}"
        async with connect(
            uri,
            origin=None,
            compression=None,
            proxy=None,
            open_timeout=2,
            close_timeout=1,
            max_size=4 * 1024,
            max_queue=4,
        ) as websocket:
            await authenticate_async_client(
                websocket,
                _voice_auth_key(config),
                AUTH_PROBE_FRAME,
            )
            probe_result = await asyncio.wait_for(websocket.recv(), timeout=2.0)
            if probe_result != AUTH_PROBE_SUCCESS_FRAME:
                return False
        return True
    except Exception:
        return False


def _status(stack: PrivateStackConfig) -> dict[str, Any]:
    config = OrchestratorConfig.load(stack.orchestrator_config_path)
    inference_ready = False
    inference_identity_ready = False
    identity_endpoint_missing = False
    try:
        client = InferenceControlClient(
            config.inference_base_url,
            config.inference_bearer_token,
            timeout_seconds=2,
        )
        capability = client.capabilities()
        inference_ready = capability.text and capability.stream
    except Exception:
        inference_ready = False
    if inference_ready:
        try:
            live_identity = client.runtime_identity()
            expected_manifest, expected_runtime = (
                stack.expected_inference_runtime_identity()
            )
            inference_identity_ready = (
                live_identity.manifest_digest == expected_manifest
                and live_identity.runtime_identity_sha256 == expected_runtime
            )
        except InferenceIdentityEndpointUnavailable:
            identity_endpoint_missing = True
        except Exception:
            inference_identity_ready = False
    foreground_session_ready = bool(
        identity_endpoint_missing and validate_live_session_lease(stack)
    )
    inference_identity_ready = (
        inference_identity_ready or foreground_session_ready
    )
    inference_identity_source = (
        "runtime_endpoint"
        if inference_identity_ready and not foreground_session_ready
        else "foreground_lease"
        if foreground_session_ready
        else "none"
    )
    orchestrator_ready = _readyz(config)
    voice_auth_ready = asyncio.run(_voice_auth_ready(config))
    beta_session = _beta_session(config)
    voice_client_connected = bool(
        beta_session and beta_session["voice_client_connected"]
    )
    services_ready = (
        inference_ready
        and inference_identity_ready
        and orchestrator_ready
        and voice_auth_ready
        and bool(beta_session and beta_session["core_ready"])
    )
    return {
        "config_valid": True,
        "foreground_session_ready": foreground_session_ready,
        "inference_identity_ready": inference_identity_ready,
        "inference_identity_source": inference_identity_source,
        "inference_ready": inference_ready,
        "orchestrator_ready": orchestrator_ready,
        "services_ready": services_ready,
        "voice_auth_ready": voice_auth_ready,
        "voice_client_connected": voice_client_connected,
        "ready": services_ready and voice_client_connected,
        "schema_version": STACK_SCHEMA_VERSION,
        "version": "jarvis-stack-status/5",
    }


def main(argv: list[str] | None = None) -> int:
    parsed = _parser().parse_args(argv)
    try:
        if parsed.command == "prepare-auth-v2":
            from .stack_migration import prepare_auth_v2_stack

            result = prepare_auth_v2_stack(
                parsed.legacy_config,
                parsed.output_directory,
                macos_ptt_key=parsed.macos_ptt_key,
                safe_beta_scope=parsed.safe_beta_scope,
                rotate_weak_bearer_tokens=parsed.rotate_weak_bearer_tokens,
            )
            _print(result.canonical())
            return 0
        if parsed.command == "init":
            document = {
                "schema_version": STACK_SCHEMA_VERSION,
                "voice_config_path": str(parsed.voice_config.expanduser().resolve()),
                "orchestrator_config_path": str(
                    parsed.orchestrator_config.expanduser().resolve()
                ),
                "inference": {
                    "runner_path": str(parsed.inference_runner.expanduser().resolve()),
                    "manifest_path": str(
                        parsed.inference_manifest.expanduser().resolve()
                    ),
                    "qualification_report_path": str(
                        parsed.inference_qualification_report.expanduser().resolve()
                    ),
                },
            }
            write_private_stack_config(parsed.output, document)
            stack = PrivateStackConfig.load(parsed.output)
            _print(
                {
                    "config_sha256": stack.sha256,
                    "ready": True,
                    "schema_version": STACK_SCHEMA_VERSION,
                }
            )
            return 0
        stack = _load(parsed)
        if parsed.command == "validate":
            _print(
                {
                    "config_sha256": stack.sha256,
                    "ready": True,
                    "schema_version": STACK_SCHEMA_VERSION,
                }
            )
            return 0
        if parsed.command == "beta-preflight":
            from .beta_preflight import evaluate_macos_beta_preflight

            report = evaluate_macos_beta_preflight(stack)
            if parsed.output is not None:
                write_readiness_report(parsed.output, report)
            _print(report)
            return 0 if report["technical_beta_preflight_ready"] else 1
        if parsed.command == "beta-run":
            from .macos_beta_session import CommandPrefix, MacOSBetaSession

            runtime_python = parsed.runtime_python.expanduser().absolute()
            session = MacOSBetaSession(
                stack_path=stack.source_path,
                orchestrator_command=CommandPrefix(
                    runtime_python, ("-m", "jarvis_orchestrator")
                ),
                voice_command=CommandPrefix(
                    runtime_python, ("-m", "jarvis_voice")
                ),
                session_parent=parsed.session_parent.expanduser().absolute(),
                startup_timeout_seconds=parsed.startup_timeout_seconds,
            )
            report = session.run().canonical()
            _print(report)
            return 0
        if parsed.command == "beta-recover":
            from .macos_beta_session import recover_macos_beta_session

            report = recover_macos_beta_session(
                stack_path=stack.source_path,
                session_parent=parsed.session_parent.expanduser().absolute(),
            ).canonical()
            _print(report)
            return 0
        if parsed.command == "physical-turn":
            from .physical_turn_gate import run_macos_physical_turn_gate

            report = run_macos_physical_turn_gate(
                stack,
                timeout_seconds=parsed.timeout_seconds,
                poll_interval_seconds=parsed.poll_interval_seconds,
            )
            if parsed.output is not None:
                write_readiness_report(parsed.output, report)
            _print(report)
            return 0 if report["ready"] else 1
        if parsed.command == "status":
            status = _status(stack)
            _print(status)
            return 0 if status["ready"] else 1
        config = OrchestratorConfig.load(stack.orchestrator_config_path)
        if parsed.command == "model-failure":
            report = asyncio.run(
                run_live_model_failure_gate(
                    host=config.voice_host,
                    port=config.voice_port,
                    auth_key=_voice_auth_key(config),
                    timeout_seconds=parsed.timeout_seconds,
                )
            )
            report = {
                **report,
                "config_sha256": stack.sha256,
                "configured_model_timeout_seconds": config.model_timeout_seconds,
            }
            if parsed.output is not None:
                write_readiness_report(parsed.output, report)
            _print(report)
            return 0 if report["ready"] else 1
        status = _status(stack)
        if not status["ready"]:
            raise StackConfigError("stack_not_ready")
        report = asyncio.run(
            run_live_sequential_gate(
                host=config.voice_host,
                port=config.voice_port,
                auth_key=_voice_auth_key(config),
                turns=parsed.turns,
                database_path=config.database_path,
            maximum_database_bytes=512 * 1024 * 1024,
            request_text="Reply exactly: CYRUS local E2E ready.",
            include_latency_profile=True,
        )
        )
        report = {
            **report,
            "config_sha256": stack.sha256,
            "real_local_model": True,
            "version": "jarvis-private-stack-e2e/1",
        }
        if parsed.output is not None:
            write_readiness_report(parsed.output, report)
        _print(report)
        return 0 if report["ready"] else 1
    except StackConfigError as exc:
        _print({"error": {"code": exc.code, "retryable": False}, "ready": False})
        return 2
    except Exception as exc:
        from .macos_beta_session import MacOSBetaSessionError

        if isinstance(exc, MacOSBetaSessionError):
            preflight_report = None
            if exc.code == "beta_preflight_failed" and parsed.command == "beta-run":
                try:
                    from .beta_preflight import evaluate_macos_beta_preflight

                    preflight_report = evaluate_macos_beta_preflight(stack)
                except Exception:
                    preflight_report = None
            _print(
                _beta_session_error_payload(
                    exc.code,
                    preflight_report=preflight_report,
                )
            )
            return 2
        _print(
            {
                "error": {"code": "stack_operation_failed", "retryable": False},
                "ready": False,
            }
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
