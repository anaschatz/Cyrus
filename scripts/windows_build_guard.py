"""Offline guard and seal operations for the native Windows build."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src"
INFERENCE_SOURCE = ROOT / "inference" / "src"
for source_root in (SOURCE, INFERENCE_SOURCE):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from jarvis_inference.private_temp import (
    PrivateTempFailure,
    private_temp_environment,
    trusted_windows_directory,
    validate_private_temp_directory,
)
from jarvis_inference.windows_process import (
    WindowsProcessFailure,
    run_windows_job,
)

from jarvis_voice.windows.cuda_bundle import (
    CudaBundleError,
    build_cuda_manifest,
    canonical_json_bytes,
    create_runner_manifest,
    load_cuda_bundle,
    require_native_windows_path,
    verify_runner_manifest,
)


def _write_exclusive(path: Path, payload: dict[str, object]) -> None:
    parent = require_native_windows_path(path.parent, "directory")
    target = parent / path.name
    if target.exists():
        raise CudaBundleError("runner_bundle.output_exists")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY
    descriptor = os.open(target, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(canonical_json_bytes(payload))
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            target.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="windows-build-guard")
    commands = parser.add_subparsers(dest="command", required=True)

    cuda = commands.add_parser("verify-cuda")
    cuda.add_argument("--manifest", type=Path, required=True)

    create_cuda = commands.add_parser("create-cuda-manifest")
    create_cuda.add_argument("--root", type=Path, required=True)
    create_cuda.add_argument("--output", type=Path, required=True)

    inputs = commands.add_parser("verify-inputs")
    inputs.add_argument("--file", action="append", default=[], type=Path)
    inputs.add_argument("--directory", action="append", default=[], type=Path)

    seal = commands.add_parser("seal-runners")
    seal.add_argument("--directory", type=Path, required=True)
    seal.add_argument("--cuda-manifest", type=Path, required=True)
    seal.add_argument("--output", type=Path, required=True)

    verify = commands.add_parser("verify-runners")
    verify.add_argument("--manifest", type=Path, required=True)

    smoke = commands.add_parser("boot-smoke")
    smoke.add_argument("--runner", action="append", required=True, type=Path)
    smoke.add_argument("--temporary-directory", required=True, type=Path)
    return parser


def main(arguments: list[str] | None = None) -> int:
    args: argparse.Namespace | None = None
    try:
        args = _parser().parse_args(arguments)
        if args.command == "verify-cuda":
            bundle = load_cuda_bundle(args.manifest)
            result = {
                "ready": True,
                "schema": "cyrus.windows-build-guard-result/1",
                "cuda_runtime_manifest_sha256": bundle.manifest_sha256,
                "dll_count": len(bundle.files),
            }
        elif args.command == "create-cuda-manifest":
            root = require_native_windows_path(args.root, "directory")
            output_parent = require_native_windows_path(
                args.output.parent, "directory"
            )
            try:
                if output_parent.samefile(root):
                    raise CudaBundleError("cuda_bundle.manifest_inside_bundle")
            except OSError as exc:
                raise CudaBundleError("cuda_bundle.path_unreadable") from exc
            payload = build_cuda_manifest(root)
            _write_exclusive(args.output, payload)
            bundle = load_cuda_bundle(args.output)
            result = {
                "ready": True,
                "schema": "cyrus.windows-build-guard-result/1",
                "cuda_runtime_manifest_sha256": bundle.manifest_sha256,
                "dll_count": len(bundle.files),
            }
        elif args.command == "verify-inputs":
            for path in args.file:
                require_native_windows_path(path, "file")
            for path in args.directory:
                require_native_windows_path(path, "directory")
            result = {
                "ready": True,
                "schema": "cyrus.windows-build-guard-result/1",
                "file_count": len(args.file),
                "directory_count": len(args.directory),
            }
        elif args.command == "seal-runners":
            directory = require_native_windows_path(args.directory, "directory")
            output_parent = require_native_windows_path(
                args.output.parent, "directory"
            )
            try:
                if not output_parent.samefile(directory):
                    raise CudaBundleError("runner_bundle.manifest_not_adjacent")
            except OSError as exc:
                raise CudaBundleError("runner_bundle.path_unreadable") from exc
            bundle = load_cuda_bundle(args.cuda_manifest)
            payload = create_runner_manifest(directory, bundle.manifest_sha256)
            _write_exclusive(args.output, payload)
            result = {
                "ready": True,
                "schema": "cyrus.windows-build-guard-result/1",
                "runner_count": len(payload["files"]),
            }
        elif args.command == "verify-runners":
            payload = verify_runner_manifest(args.manifest)
            result = {
                "ready": True,
                "schema": "cyrus.windows-build-guard-result/1",
                "runner_count": len(payload["files"]),
            }
        else:
            temporary = validate_private_temp_directory(args.temporary_directory)
            windows = trusted_windows_directory()
            environment = {
                "SystemRoot": str(windows),
                "WINDIR": str(windows),
                "OS": "Windows_NT",
                **private_temp_environment(temporary),
            }
            for runner_value in args.runner:
                runner = require_native_windows_path(runner_value, "file")
                completed = run_windows_job(
                    [str(runner)],
                    environment=environment,
                    cwd=runner.parent,
                    timeout=30.0,
                )
                if completed.returncode != 2:
                    raise CudaBundleError("runner_bundle.boot_smoke_failed")
            result = {
                "ready": True,
                "schema": "cyrus.windows-build-guard-result/1",
                "runner_count": len(args.runner),
            }
    except (
        CudaBundleError,
        OSError,
        PrivateTempFailure,
        subprocess.TimeoutExpired,
        ValueError,
        WindowsProcessFailure,
    ) as exc:
        if args is not None and args.command == "boot-smoke":
            code = "runner_bundle.boot_smoke_failed"
        elif isinstance(exc, CudaBundleError):
            code = exc.code
        elif isinstance(
            exc,
            (PrivateTempFailure, subprocess.TimeoutExpired, WindowsProcessFailure),
        ):
            code = "runner_bundle.boot_smoke_failed"
        else:
            code = "windows_build.failed"
        print(
            json.dumps(
                {
                    "ready": False,
                    "schema": "cyrus.windows-build-guard-result/1",
                    "code": code,
                },
                separators=(",", ":"),
                sort_keys=True,
            )
        )
        return 2
    print(json.dumps(result, separators=(",", ":"), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
