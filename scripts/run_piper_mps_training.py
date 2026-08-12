#!/usr/bin/env python3
"""Inspect or describe native M3/MPS STOMA F1 training."""

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
from jarvis_voice.piper_mps_training import (  # noqa: E402
    piper_mps_training_commands,
    piper_mps_training_readiness,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    readiness = commands.add_parser("readiness")
    readiness.add_argument("--data-dir", type=Path, required=True)
    readiness.add_argument("--artifact-root", type=Path, required=True)
    readiness.add_argument("--work-dir", type=Path, required=True)
    readiness.add_argument("--source-root", type=Path)
    plan = commands.add_parser("plan")
    plan.add_argument("--python", type=Path, required=True)
    plan.add_argument("--data-dir", type=Path, required=True)
    plan.add_argument("--artifact-root", type=Path, required=True)
    plan.add_argument("--work-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "readiness":
            report = piper_mps_training_readiness(
                args.data_dir.resolve(),
                args.artifact_root.resolve(),
                args.work_dir.resolve(),
                source_root=None if args.source_root is None else args.source_root.resolve(),
            )
        else:
            report = piper_mps_training_commands(
                args.python.absolute(),
                args.data_dir.resolve(),
                args.artifact_root.resolve(),
                args.work_dir.resolve(),
            )
    except BackendError as exc:
        report = {
            "ready": False,
            "code": exc.code,
            "network_used": False,
            "installation_performed": False,
            "training_started": False,
        }
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report.get("ready", False) else 2


if __name__ == "__main__":
    raise SystemExit(main())
