#!/usr/bin/env python3
"""Build or verify the offline STOMA F1 WSL training handoff."""

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
from jarvis_voice.piper_training_handoff import (  # noqa: E402
    build_piper_f1_handoff,
    verify_piper_f1_handoff,
    write_piper_f1_handoff,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    build.add_argument("--data-dir", type=Path, required=True)
    build.add_argument("--artifact-root", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("--transfer-root", type=Path, required=True)
    verify.add_argument("--manifest", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "build":
            manifest = build_piper_f1_handoff(args.data_dir.resolve(), args.artifact_root.resolve())
            write_piper_f1_handoff(args.output.resolve(), manifest)
            report = {
                "ready": True,
                "code": "tts.f1_handoff.created",
                "files": manifest["totals"]["files"],
                "bytes": manifest["totals"]["bytes"],
                "network_used": False,
                "installation_performed": False,
                "training_started": False,
            }
        else:
            report = verify_piper_f1_handoff(args.transfer_root.resolve(), args.manifest.resolve())
    except BackendError as exc:
        report = {
            "ready": False,
            "code": exc.code,
            "network_used": False,
            "installation_performed": False,
            "training_started": False,
        }
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
