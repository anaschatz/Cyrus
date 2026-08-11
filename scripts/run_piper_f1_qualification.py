#!/usr/bin/env python3
"""Preflight or qualify a local STOMA F1 Piper ONNX candidate on macOS."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from jarvis_voice.errors import BackendError  # noqa: E402
from jarvis_voice.macos.piper_f1_qualification import (  # noqa: E402
    SoundDevicePlaybackSink,
    piper_f1_qualification_preflight,
    run_piper_f1_qualification,
    write_qualification_report,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight = subparsers.add_parser("preflight", help="read-only readiness check")
    preflight.add_argument("--candidate-root", type=Path, required=True)

    qualify = subparsers.add_parser("qualify", help="generate fixtures and measure the candidate")
    qualify.add_argument("--candidate-root", type=Path, required=True)
    qualify.add_argument("--output-directory", type=Path, required=True)
    qualify.add_argument("--report", type=Path)
    qualify.add_argument("--exercise-playback-cancel", action="store_true")
    qualify.add_argument("--human-voice-approved", action="store_true")
    args = parser.parse_args(argv)

    try:
        if args.command == "preflight":
            report = piper_f1_qualification_preflight(args.candidate_root.resolve())
        else:
            sink_factory = SoundDevicePlaybackSink if args.exercise_playback_cancel else None
            report = run_piper_f1_qualification(
                args.candidate_root.resolve(),
                args.output_directory.resolve(),
                playback_sink_factory=sink_factory,
                human_voice_approved=args.human_voice_approved,
            )
            if args.report is not None:
                write_qualification_report(args.report.resolve(), report)
    except BackendError as exc:
        report = {
            "ready": False,
            "code": exc.code,
            "network_used": False,
            "installation_performed": False,
            "production_backend_changed": False,
        }
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report.get("ready_for_promotion", report.get("ready", False)) else 2


if __name__ == "__main__":
    raise SystemExit(main())
