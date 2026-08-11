#!/usr/bin/env python3
"""Plan, acquire, extract, or describe training for the STOMA F1 pilot."""

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
from jarvis_voice.macos.compact_greek_tts_acquisition import (  # noqa: E402
    acquire_pinned_artifacts,
    compact_acquisition_plan,
    compact_f1_training_plan,
    extract_stoma_f1,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "acquire"):
        command = commands.add_parser(name)
        command.add_argument("--storage-root", type=Path, required=True)
        command.add_argument("--group", choices=("all", "stoma", "checkpoint", "runtime"), default="all")
        if name == "acquire":
            command.add_argument("--confirm-download-bytes", type=int, required=True)
    extract = commands.add_parser("extract")
    extract.add_argument("--artifact-root", type=Path, required=True)
    extract.add_argument("--output-dir", type=Path, required=True)
    training = commands.add_parser("training-plan")
    training.add_argument("--data-dir", type=Path, required=True)
    training.add_argument("--artifact-root", type=Path, required=True)
    training.add_argument("--work-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "plan":
            report = compact_acquisition_plan(args.storage_root.resolve(), group=args.group)
        elif args.command == "acquire":
            report = acquire_pinned_artifacts(
                args.storage_root.resolve(),
                group=args.group,
                confirm_download_bytes=args.confirm_download_bytes,
            )
        elif args.command == "extract":
            report = extract_stoma_f1(args.artifact_root.resolve(), args.output_dir.resolve())
        else:
            report = compact_f1_training_plan(
                args.data_dir.resolve(),
                args.artifact_root.resolve(),
                args.work_dir.resolve(),
            )
    except BackendError as exc:
        report = {"ready": False, "code": exc.code}
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
