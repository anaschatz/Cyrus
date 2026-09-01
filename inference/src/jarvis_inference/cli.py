from __future__ import annotations

import argparse
import json
import os
import signal
import sys
from pathlib import Path
from typing import Any

from .errors import ApplianceError
from .gateway import Production, serve
from .gateway import effective_configuration, package_digest
from .llama_engine import LlamaServerEngine
from .manifest import (
    Manifest,
    assert_runtime_platform,
    hash_file,
    load_manifest,
    resolve_local,
    runtime_identity,
    verify_artifacts,
)
from .mlx_engine import MlxWorkerEngine
from .durable import read_report
from .primary_qualification import qualify_primary_profile
from .profile_report import (
    ATTESTATION_LEVEL,
    OPERATOR_TRUST_REQUIRED,
    profile_report_matches,
)
from .qualification import qualify
from .circuit import RestartCircuitBreaker

_LLAMA_BREAKER = RestartCircuitBreaker()
_MLX_BREAKER = RestartCircuitBreaker()


def engine_factory(manifest: Manifest):
    if manifest.platform == "windows-x86_64":
        return LlamaServerEngine(manifest, _LLAMA_BREAKER)
    return MlxWorkerEngine(manifest, _MLX_BREAKER)


def qualification_engine_factory(manifest: Manifest):
    # Every intentionally injected native fault is isolated from the production
    # rolling restart circuit. The production factory above keeps one breaker.
    breaker = RestartCircuitBreaker()
    if manifest.platform == "windows-x86_64":
        return LlamaServerEngine(manifest, breaker)
    return MlxWorkerEngine(manifest, breaker)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cyrus-inference")
    commands = parser.add_subparsers(dest="command", required=True)
    serve_parser = commands.add_parser("serve", help="serve the qualified appliance")
    serve_parser.add_argument("--manifest", type=Path, required=True)
    qualify_parser = commands.add_parser("qualify", help="write a native qualification report")
    qualify_parser.add_argument("--manifest", type=Path, required=True)
    qualify_parser.add_argument("--fixture-runner", type=Path, required=True)
    profile_parser = commands.add_parser(
        "qualify-profile",
        help="write the exact Windows local-primary profile qualification report",
    )
    profile_parser.add_argument("--manifest", type=Path, required=True)
    profile_parser.add_argument("--fixture-runner", type=Path, required=True)
    verify_parser = commands.add_parser("verify-manifest", help="strictly verify the manifest and artifacts")
    verify_parser.add_argument("--manifest", type=Path, required=True)
    readiness_parser = commands.add_parser(
        "verify-readiness",
        help="fail fast unless every production qualification report is current",
    )
    readiness_parser.add_argument("--manifest", type=Path, required=True)
    return parser


def _generic_report_matches(
    manifest: Manifest, identity: dict[str, Any], report: Any
) -> bool:
    if (
        not isinstance(report, dict)
        or report.get("report_schema_version") != 1
        or report.get("status") != "pass"
        or report.get("identity") != identity
    ):
        return False
    runner = report.get("fixture_runner")
    if not isinstance(runner, dict) or set(runner) != {
        "path",
        "size_bytes",
        "sha256",
    }:
        return False
    try:
        path = resolve_local(manifest.root, runner["path"], "fixture_runner")
        size, digest = hash_file(path)
    except (ApplianceError, OSError, TypeError):
        return False
    return (
        type(runner["size_bytes"]) is int
        and runner["size_bytes"] == size
        and runner["sha256"] == digest
    )


def verify_readiness(manifest: Manifest) -> dict[str, Any]:
    assert_runtime_platform(manifest)
    verify_artifacts(manifest)
    identity = runtime_identity(
        manifest, package_digest(), effective_configuration()
    )
    report = read_report(manifest.qualification_report)
    if not _generic_report_matches(manifest, identity, report):
        raise ApplianceError(
            503,
            "qualification_not_ready",
            "The generic qualification report is missing, stale, or invalid.",
        )
    assert isinstance(report, dict)
    if not profile_report_matches(manifest, identity, report):
        raise ApplianceError(
            503,
            "profile_qualification_not_ready",
            "The model profile qualification report is missing, stale, or invalid.",
        )
    return {
        "production_ready": True,
        "manifest_digest": manifest.digest,
        "attestation_level": ATTESTATION_LEVEL,
        "operator_trust_required": OPERATOR_TRUST_REQUIRED,
    }


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        manifest = load_manifest(args.manifest)
        if args.command == "verify-manifest":
            verify_artifacts(manifest)
            print(json.dumps({"valid": True, "manifest_digest": manifest.digest}, separators=(",", ":")))
            return 0
        if args.command == "verify-readiness":
            print(json.dumps(verify_readiness(manifest), separators=(",", ":")))
            return 0
        if args.command == "qualify":
            report = qualify(manifest, qualification_engine_factory, args.fixture_runner)
            print(json.dumps({"status": report["status"], "report_written": True}, separators=(",", ":")))
            return 0
        if args.command == "qualify-profile":
            report = qualify_primary_profile(
                manifest, qualification_engine_factory, args.fixture_runner
            )
            print(
                json.dumps(
                    {
                        "status": report["status"],
                        "profile_report_written": True,
                    },
                    separators=(",", ":"),
                )
            )
            return 0
        assert_runtime_platform(manifest)
        secret = os.environ.get("JARVIS_INFERENCE_API_KEY", "")
        if len(secret.encode("utf-8")) < 32:
            raise ApplianceError(503, "missing_api_key", "JARVIS_INFERENCE_API_KEY must contain at least 32 UTF-8 bytes.")
        production = Production(manifest, secret, engine_factory)
        previous_term = signal.getsignal(signal.SIGTERM)

        def stop_service(signum: int, frame: Any) -> None:
            raise KeyboardInterrupt

        signal.signal(signal.SIGTERM, stop_service)
        try:
            serve(production)
        except KeyboardInterrupt:
            return 0
        finally:
            signal.signal(signal.SIGTERM, previous_term)
        return 0
    except ApplianceError as exc:
        print(json.dumps(exc.public(), separators=(",", ":")), file=sys.stderr)
        return 2
    except (OSError, ValueError) as exc:
        print(json.dumps({"error": {"type": "server_error", "code": "startup_failed", "message": "The inference appliance could not start."}}, separators=(",", ":")), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
