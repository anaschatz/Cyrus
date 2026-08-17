#!/usr/bin/env python3
"""Run bounded, evidence-producing acoustic qualification for the approved F1 voice."""

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

from jarvis_voice.config import VoiceConfig, load_voice_config  # noqa: E402
from jarvis_voice.errors import BackendError  # noqa: E402
from jarvis_voice.macos.native_run import (  # noqa: E402
    calibrate_macos_acoustic_threshold,
    run_macos_acoustic_barge_in_sample,
)
from jarvis_voice.qualification import write_report  # noqa: E402


MAXIMUM_SAMPLES_PER_PROCESS = 2


def _bounded_sample_count(value: str) -> int:
    count = int(value)
    if not 1 <= count <= MAXIMUM_SAMPLES_PER_PROCESS:
        raise argparse.ArgumentTypeError("sample count must be 1 or 2")
    return count


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--approval-manifest", type=Path, required=True)
    parser.add_argument("--approval-manifest-sha256", required=True)
    parser.add_argument("--repository-root", type=Path, default=ROOT)
    parser.add_argument("--microphone", default="MacBook Air Microphone")
    parser.add_argument("--ptt-key", default="CONTROL_OPTION")
    commands = parser.add_subparsers(dest="command", required=True)

    calibrate = commands.add_parser("calibrate")
    calibrate.add_argument("--output", type=Path, required=True)
    calibrate.add_argument("--ambient-seconds", type=float, default=5.0)

    batch = commands.add_parser("batch")
    batch.add_argument("--output", type=Path, required=True)
    batch.add_argument("--silence-threshold-dbfs", type=float, required=True)
    batch.add_argument("--sample-count", type=_bounded_sample_count, default=2)
    batch.add_argument("--ptt-timeout-seconds", type=float, default=60.0)
    return parser


def qualification_config(args: argparse.Namespace) -> VoiceConfig:
    config = load_voice_config(args.config)
    greek = replace(
        config.macos.greektts15,
        model_root=str(args.model_root.resolve()),
        approval_manifest_path=str(args.approval_manifest.resolve()),
        approval_manifest_sha256=args.approval_manifest_sha256,
        repository_root=str(args.repository_root.resolve()),
        warm_on_start=True,
    )
    return replace(
        config,
        ptt_key=args.ptt_key,
        audio_input_device=args.microphone,
        macos=replace(config.macos, tts_backend="greektts15_f1", greektts15=greek),
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = qualification_config(args)
        if args.command == "calibrate":
            print("Calibration started. Keep the room quiet until the test phrase plays.", flush=True)
            report = calibrate_macos_acoustic_threshold(
                config,
                audio_path="built_in",
                ambient_seconds=args.ambient_seconds,
            )
        else:
            print(
                f"Bounded acoustic batch started ({args.sample_count}/{MAXIMUM_SAMPLES_PER_PROCESS}). "
                "Press the configured PTT chord during each spoken phrase.",
                flush=True,
            )
            report = run_macos_acoustic_barge_in_sample(
                config,
                silence_threshold_dbfs=args.silence_threshold_dbfs,
                audio_path="built_in",
                ptt_timeout_seconds=args.ptt_timeout_seconds,
                sample_count=args.sample_count,
                progress_sink=lambda value: print(json.dumps(value, ensure_ascii=False, sort_keys=True), flush=True),
            )
        report = {**report, "maximum_samples_per_process": MAXIMUM_SAMPLES_PER_PROCESS}
    except BackendError as exc:
        report = {
            "ready": False,
            "code": exc.code,
            "maximum_samples_per_process": MAXIMUM_SAMPLES_PER_PROCESS,
            "network_used": False,
            "installation_performed": False,
        }
        if exc.safe_context:
            report["context"] = dict(exc.safe_context)
    write_report(args.output.resolve(), report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report.get("ready") else 2


if __name__ == "__main__":
    raise SystemExit(main())
