from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from .errors import ApplianceError
from .gateway import Production, serve
from .llama_engine import LlamaServerEngine
from .manifest import Manifest, assert_runtime_platform, load_manifest, verify_artifacts
from .mlx_engine import MlxWorkerEngine
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
    parser = argparse.ArgumentParser(prog="jarvis-inference")
    commands = parser.add_subparsers(dest="command", required=True)
    serve_parser = commands.add_parser("serve", help="serve the qualified appliance")
    serve_parser.add_argument("--manifest", type=Path, required=True)
    qualify_parser = commands.add_parser("qualify", help="write a native qualification report")
    qualify_parser.add_argument("--manifest", type=Path, required=True)
    qualify_parser.add_argument("--fixture-runner", type=Path, required=True)
    verify_parser = commands.add_parser("verify-manifest", help="strictly verify the manifest and artifacts")
    verify_parser.add_argument("--manifest", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        manifest = load_manifest(args.manifest)
        if args.command == "verify-manifest":
            verify_artifacts(manifest)
            print(json.dumps({"valid": True, "manifest_digest": manifest.digest}, separators=(",", ":")))
            return 0
        if args.command == "qualify":
            report = qualify(manifest, qualification_engine_factory, args.fixture_runner)
            print(json.dumps({"status": report["status"], "report_written": True}, separators=(",", ":")))
            return 0
        assert_runtime_platform(manifest)
        secret = os.environ.get("JARVIS_INFERENCE_API_KEY", "")
        if len(secret.encode("utf-8")) < 32:
            raise ApplianceError(503, "missing_api_key", "JARVIS_INFERENCE_API_KEY must contain at least 32 UTF-8 bytes.")
        production = Production(manifest, secret, engine_factory)
        serve(production)
        return 0
    except ApplianceError as exc:
        print(json.dumps(exc.public(), separators=(",", ":")), file=sys.stderr)
        return 2
    except (OSError, ValueError) as exc:
        print(json.dumps({"error": {"type": "server_error", "code": "startup_failed", "message": "The inference appliance could not start."}}, separators=(",", ":")), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
