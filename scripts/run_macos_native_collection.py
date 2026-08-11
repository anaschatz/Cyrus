#!/usr/bin/env python3
"""Preflight or run the real macOS native voice telemetry collector."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from jarvis_voice.config import load_voice_config  # noqa: E402
from jarvis_voice.errors import BackendError  # noqa: E402
from jarvis_voice.macos.native_run import (  # noqa: E402
    calibrate_macos_acoustic_threshold,
    macos_native_collection_preflight,
    run_macos_native_collection,
)
from jarvis_voice.qualification import write_report  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("preflight", "run"):
        command = commands.add_parser(name)
        command.add_argument("--config", type=Path, required=True)
        command.add_argument("--audio-path", choices=("wired", "built_in"), required=True)
        command.add_argument("--silence-threshold-dbfs", type=float, required=True)
        command.add_argument("--threshold-calibrated", action="store_true")
    preflight = commands.choices["preflight"]
    preflight.add_argument("--output", type=Path)
    run = commands.choices["run"]
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--duration-seconds", type=float, default=28_800.0)
    run.add_argument("--rss-interval-seconds", type=float, default=300.0)
    calibrate = commands.add_parser("calibrate")
    calibrate.add_argument("--config", type=Path, required=True)
    calibrate.add_argument("--audio-path", choices=("wired", "built_in"), required=True)
    calibrate.add_argument("--output", type=Path, required=True)
    calibrate.add_argument("--ambient-seconds", type=float, default=3.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = load_voice_config(args.config)
        if args.command == "calibrate":
            print("Acoustic calibration started. Keep the room quiet until the test phrase plays.", flush=True)
            report = calibrate_macos_acoustic_threshold(
                config,
                audio_path=args.audio_path,
                ambient_seconds=args.ambient_seconds,
            )
            write_report(args.output.resolve(), report)
        elif args.command == "preflight":
            report = macos_native_collection_preflight(
                config,
                silence_threshold_dbfs=args.silence_threshold_dbfs,
                audio_path=args.audio_path,
                threshold_calibrated=args.threshold_calibrated,
            )
            if args.output is not None:
                write_report(args.output.resolve(), report)
        else:
            print(
                "Native collection started. Use physical PTT normally; press Ctrl-C for a blocked partial report.",
                flush=True,
            )
            report = run_macos_native_collection(
                config,
                output=args.output.resolve(),
                silence_threshold_dbfs=args.silence_threshold_dbfs,
                audio_path=args.audio_path,
                duration_seconds=args.duration_seconds,
                rss_interval_seconds=args.rss_interval_seconds,
                threshold_calibrated=args.threshold_calibrated,
                progress_sink=lambda value: print(json.dumps(value, sort_keys=True), flush=True),
            )
    except BackendError as exc:
        report = {"ready": False, "code": exc.code, "network_used": False, "installation_performed": False}
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report.get("ready") else 2


if __name__ == "__main__":
    raise SystemExit(main())
