#!/usr/bin/env python3
"""Preflight or run the real macOS native voice telemetry collector."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from jarvis_voice.config import ArtifactConfig, VoiceConfig, load_voice_config  # noqa: E402
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
        command.add_argument("--f1-worker-python", type=Path)
        command.add_argument("--f1-worker-python-size", type=int)
        command.add_argument("--f1-worker-python-sha256")
        command.add_argument("--f1-model-root", type=Path)
        command.add_argument("--f1-approval-manifest", type=Path)
        command.add_argument("--f1-approval-manifest-sha256")
        command.add_argument("--f1-repository-root", type=Path)
    preflight = commands.choices["preflight"]
    preflight.add_argument("--output", type=Path)
    run = commands.choices["run"]
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--duration-seconds", type=float, default=28_800.0)
    run.add_argument("--rss-interval-seconds", type=float, default=300.0)
    run.add_argument(
        "--maximum-real-voice-turns",
        type=int,
        choices=range(1, 6),
        help="Stop after 1-5 transcripts have also completed playback.",
    )
    calibrate = commands.add_parser("calibrate")
    calibrate.add_argument("--config", type=Path, required=True)
    calibrate.add_argument("--audio-path", choices=("wired", "built_in"), required=True)
    calibrate.add_argument("--output", type=Path, required=True)
    calibrate.add_argument("--ambient-seconds", type=float, default=3.0)
    return parser


def apply_f1_sidecar_profile(config: VoiceConfig, args: argparse.Namespace) -> VoiceConfig:
    values = (
        args.f1_worker_python,
        args.f1_worker_python_size,
        args.f1_worker_python_sha256,
        args.f1_model_root,
        args.f1_approval_manifest,
        args.f1_approval_manifest_sha256,
        args.f1_repository_root,
    )
    if all(value is None for value in values):
        return config
    if any(value is None for value in values):
        raise BackendError(
            "qualification.f1_sidecar_profile_incomplete",
            "F1 sidecar qualification profile is incomplete",
            subsystem="qualification",
        )
    greek = replace(
        config.macos.greektts15,
        model_root=str(args.f1_model_root.resolve()),
        approval_manifest_path=str(args.f1_approval_manifest.resolve()),
        approval_manifest_sha256=args.f1_approval_manifest_sha256,
        repository_root=str(args.f1_repository_root.resolve()),
        warm_on_start=True,
        worker_python=ArtifactConfig(
            str(args.f1_worker_python.expanduser().absolute()),
            args.f1_worker_python_size,
            args.f1_worker_python_sha256,
        ),
    )
    return replace(config, macos=replace(config.macos, tts_backend="greektts15_f1", greektts15=greek))


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = load_voice_config(args.config)
        if args.command in {"preflight", "run"}:
            config = apply_f1_sidecar_profile(config, args)
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
            if args.maximum_real_voice_turns is None:
                message = "Native collection started. Use physical PTT normally; press Ctrl-C for a blocked partial report."
            else:
                message = (
                    f"Bounded native batch started. Complete {args.maximum_real_voice_turns} physical PTT turn(s); "
                    "the process exits after their playback completes."
                )
            print(message, flush=True)
            report = run_macos_native_collection(
                config,
                output=args.output.resolve(),
                silence_threshold_dbfs=args.silence_threshold_dbfs,
                audio_path=args.audio_path,
                duration_seconds=args.duration_seconds,
                rss_interval_seconds=args.rss_interval_seconds,
                maximum_real_voice_turns=args.maximum_real_voice_turns,
                threshold_calibrated=args.threshold_calibrated,
                progress_sink=lambda value: print(json.dumps(value, sort_keys=True), flush=True),
            )
    except BackendError as exc:
        report = {"ready": False, "code": exc.code, "network_used": False, "installation_performed": False}
        if exc.safe_context:
            report["context"] = dict(exc.safe_context)
        output = getattr(args, "output", None)
        if output is not None:
            write_report(output.resolve(), report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report.get("ready") or report.get("turn_batch_completed") else 2


if __name__ == "__main__":
    raise SystemExit(main())
