"""One content-free operator entry point for the private local JARVIS stack."""

from __future__ import annotations

import argparse
import asyncio
import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from .config import OrchestratorConfig
from .inference import InferenceControlClient
from .readiness import (
    run_live_model_failure_gate,
    run_live_sequential_gate,
    write_readiness_report,
)
from .stack_config import (
    STACK_SCHEMA_VERSION,
    PrivateStackConfig,
    StackConfigError,
    write_private_stack_config,
)


def _print(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="jarvis-stack")
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
    commands.add_parser("validate")
    commands.add_parser("status")
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


def _status(stack: PrivateStackConfig) -> dict[str, Any]:
    config = OrchestratorConfig.load(stack.orchestrator_config_path)
    inference_ready = False
    try:
        capability = InferenceControlClient(
            config.inference_base_url,
            config.inference_bearer_token,
            timeout_seconds=2,
        ).capabilities()
        inference_ready = capability.text and capability.stream
    except Exception:
        inference_ready = False
    orchestrator_ready = _readyz(config)
    return {
        "config_valid": True,
        "inference_ready": inference_ready,
        "orchestrator_ready": orchestrator_ready,
        "ready": inference_ready and orchestrator_ready,
        "schema_version": STACK_SCHEMA_VERSION,
        "version": "jarvis-stack-status/1",
    }


def main(argv: list[str] | None = None) -> int:
    parsed = _parser().parse_args(argv)
    try:
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
                turns=parsed.turns,
                database_path=config.database_path,
                maximum_database_bytes=512 * 1024 * 1024,
                request_text="Reply exactly: JARVIS local E2E ready.",
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
    except Exception:
        _print(
            {
                "error": {"code": "stack_operation_failed", "retryable": False},
                "ready": False,
            }
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
