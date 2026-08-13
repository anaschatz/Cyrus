#!/usr/bin/env python3
"""Run the pinned offline Supertonic 3 native-Greek qualification."""

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
from jarvis_voice.macos.supertonic3_pilot import (  # noqa: E402
    DEFAULT_SPEED,
    DEFAULT_TOTAL_STEPS,
    DEFAULT_VOICE,
    VOICE_NAMES,
    run_supertonic3_quality_qualification,
    score_supertonic3_outputs,
    supertonic3_pilot_preflight,
)
from jarvis_voice.qualification import write_report  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    preflight = commands.add_parser("preflight")
    preflight.add_argument("--model-dir", required=True)
    preflight.add_argument("--report")
    qualify = commands.add_parser("qualify")
    qualify.add_argument("--model-dir", required=True)
    qualify.add_argument("--output-dir", required=True)
    qualify.add_argument("--voice", choices=VOICE_NAMES, default=DEFAULT_VOICE)
    qualify.add_argument("--total-steps", type=int, default=DEFAULT_TOTAL_STEPS)
    qualify.add_argument("--speed", type=float, default=DEFAULT_SPEED)
    qualify.add_argument("--seed", type=int, default=10_000)
    qualify.add_argument("--report", required=True)
    score = commands.add_parser("score")
    score.add_argument("--output-dir", required=True)
    score.add_argument("--whisper-model-dir", required=True)
    score.add_argument("--report", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "preflight":
            report = supertonic3_pilot_preflight(args.model_dir)
        elif args.command == "qualify":
            report = run_supertonic3_quality_qualification(
                args.model_dir,
                args.output_dir,
                voice=args.voice,
                total_steps=args.total_steps,
                speed=args.speed,
                seed=args.seed,
            )
        else:
            report = score_supertonic3_outputs(args.output_dir, args.whisper_model_dir)
        if args.report:
            write_report(args.report, report)
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        return 0 if report["ready"] else 2
    except BackendError as exc:
        print(json.dumps({"ready": False, "code": exc.code}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
