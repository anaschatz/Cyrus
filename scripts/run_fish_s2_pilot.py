#!/usr/bin/env python3
"""Run the pinned, local-only Fish Audio S2 Pro Greek qualification."""

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
from jarvis_voice.macos.fish_s2_pilot import (  # noqa: E402
    QUALIFICATION_FIXTURES,
    pilot_preflight,
    run_pilot,
    score_outputs,
)
from jarvis_voice.qualification import write_report  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fixture_choices = [item[0] for item in QUALIFICATION_FIXTURES]
    for name in ("preflight", "run"):
        command = commands.add_parser(name)
        command.add_argument("--model-dir", required=True)
        command.add_argument("--reference-audio", required=True)
        command.add_argument("--runtime-source", required=True)
        command.add_argument("--report", required=True)
        if name == "run":
            command.add_argument("--output-dir", required=True)
            command.add_argument("--fixture", action="append", choices=fixture_choices)
    score = commands.add_parser("score")
    score.add_argument("--output-dir", required=True)
    score.add_argument("--whisper-model-dir", required=True)
    score.add_argument("--fixture", action="append", choices=fixture_choices)
    score.add_argument("--report", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "preflight":
            report = pilot_preflight(args.model_dir, args.reference_audio, args.runtime_source)
        elif args.command == "run":
            report = run_pilot(
                args.model_dir,
                args.output_dir,
                args.reference_audio,
                args.runtime_source,
                fixture_ids=args.fixture,
            )
        else:
            report = score_outputs(
                args.output_dir,
                args.whisper_model_dir,
                fixture_ids=args.fixture,
            )
        write_report(args.report, report)
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        return 0 if report["ready"] else 2
    except BackendError as exc:
        print(json.dumps({"ready": False, "code": exc.code}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
