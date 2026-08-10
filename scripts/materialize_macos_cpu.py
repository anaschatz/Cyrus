#!/usr/bin/env python3
"""Explicit materialization operations for the macOS CPU source environment."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from jarvis_voice.errors import BackendError
from jarvis_voice.macos.materialization import (
    blocked_materialization_summary,
    completed_materialization_summary,
    disk_space_gate,
    extract_silero_onnx,
    lock_wheelhouse,
    materialize_private_config,
    probe_local_backends,
)
from jarvis_voice.qualification import write_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="materialize-macos-cpu")
    commands = parser.add_subparsers(dest="command", required=True)
    lock = commands.add_parser("lock-wheelhouse")
    lock.add_argument("--wheelhouse", required=True)
    lock.add_argument("--output", required=True)
    silero = commands.add_parser("extract-silero")
    silero.add_argument("--wheel", required=True)
    silero.add_argument("--output-directory", required=True)
    silero.add_argument("--manifest-output", required=True)
    config = commands.add_parser("write-config")
    config.add_argument("--base-config", required=True)
    config.add_argument("--model-dir", required=True)
    config.add_argument("--vad-manifest", required=True)
    config.add_argument("--greek-voice", required=True)
    config.add_argument("--english-voice", required=True)
    config.add_argument("--output", required=True)
    probe = commands.add_parser("probe-backends")
    probe.add_argument("--config", required=True)
    probe.add_argument("--model-dir", required=True)
    probe.add_argument("--output", required=True)
    disk = commands.add_parser("disk-gate")
    disk.add_argument("--path", required=True)
    disk.add_argument("--output", required=True)
    summary = commands.add_parser("blocked-summary")
    summary.add_argument("--python-version", required=True)
    summary.add_argument("--wheels-locked", required=True, type=int)
    summary.add_argument("--silero-bytes", required=True, type=int)
    summary.add_argument("--available-mib", required=True, type=int)
    summary.add_argument("--output", required=True)
    complete = commands.add_parser("complete-summary")
    complete.add_argument("--python-version", required=True)
    complete.add_argument("--wheels-locked", required=True, type=int)
    complete.add_argument("--silero-bytes", required=True, type=int)
    complete.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report_output: str | None = None
    try:
        if args.command == "lock-wheelhouse":
            report = lock_wheelhouse(args.wheelhouse, args.output)
        elif args.command == "extract-silero":
            report = extract_silero_onnx(args.wheel, args.output_directory, args.manifest_output)
        elif args.command == "write-config":
            report = materialize_private_config(
                args.base_config,
                args.model_dir,
                args.vad_manifest,
                args.output,
                greek_voice=args.greek_voice,
                english_voice=args.english_voice,
            )
        elif args.command == "probe-backends":
            report_output = args.output
            report = probe_local_backends(args.config, args.model_dir)
            write_report(report_output, report)
        elif args.command == "disk-gate":
            report_output = args.output
            report = disk_space_gate(args.path)
            write_report(report_output, report)
        elif args.command == "blocked-summary":
            report_output = args.output
            report = blocked_materialization_summary(
                python_version=args.python_version,
                wheels_locked=args.wheels_locked,
                silero_bytes=args.silero_bytes,
                available_mib=args.available_mib,
            )
            write_report(report_output, report)
        elif args.command == "complete-summary":
            report_output = args.output
            report = completed_materialization_summary(
                python_version=args.python_version,
                wheels_locked=args.wheels_locked,
                silero_bytes=args.silero_bytes,
            )
            write_report(report_output, report)
        else:  # pragma: no cover - argparse owns the command set
            raise AssertionError("unknown command")
    except BackendError as exc:
        report = {"ready": False, "error": exc.to_log_record(), "network_used": False}
        if report_output:
            write_report(report_output, report)
        else:
            print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report.get("ready") else 2


if __name__ == "__main__":
    raise SystemExit(main())
