#!/usr/bin/env python3
"""Compile Piper's pinned monotonic-alignment extension for the M3 trainer."""

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
from jarvis_voice.piper_mps_materialization import (  # noqa: E402
    materialize_piper_mps_extension,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = materialize_piper_mps_extension(
            args.python.absolute(),
            args.source_root.absolute(),
        )
    except BackendError as exc:
        report = {
            "ready": False,
            "code": exc.code,
            "network_used": False,
            "training_started": False,
        }
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report.get("ready", False) else 2


if __name__ == "__main__":
    raise SystemExit(main())
