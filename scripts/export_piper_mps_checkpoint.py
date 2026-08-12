#!/usr/bin/env python3
"""Safely export a completed local Piper MPS checkpoint to ONNX."""

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
from jarvis_voice.piper_mps_export import run_piper_mps_export  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        run_piper_mps_export(args.checkpoint.absolute(), args.output_file.absolute())
    except BackendError as exc:
        print(json.dumps({"ready": False, "code": exc.code}, sort_keys=True))
        return 2
    print(
        json.dumps(
            {
                "ready": True,
                "code": "tts.f1_mps.export_complete",
                "output": str(args.output_file.absolute()),
                "network_used": False,
                "installation_performed": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

