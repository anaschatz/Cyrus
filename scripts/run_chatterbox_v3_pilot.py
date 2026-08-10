#!/usr/bin/env python3
"""Run the offline-first Chatterbox Multilingual V3 qualification pilot."""

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
from jarvis_voice.macos.chatterbox_pilot import (  # noqa: E402
    chatterbox_pilot_preflight,
    run_chatterbox_pilot,
)
from jarvis_voice.qualification import write_report  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    preflight = commands.add_parser("preflight")
    preflight.add_argument("--storage-root", required=True)
    preflight.add_argument("--model-dir", required=True)
    preflight.add_argument("--report")
    run = commands.add_parser("run")
    run.add_argument("--model-dir", required=True)
    run.add_argument("--output-dir", required=True)
    run.add_argument("--report", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "preflight":
            report = chatterbox_pilot_preflight(args.storage_root, args.model_dir)
        else:
            report = run_chatterbox_pilot(args.model_dir, args.output_dir)
        if args.report:
            write_report(args.report, report)
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        return 0 if report["ready"] else 2
    except BackendError as exc:
        print(json.dumps({"ready": False, "code": exc.code}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
