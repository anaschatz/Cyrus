#!/usr/bin/env python3
"""Run the passive STOMA F1 Piper training readiness gate inside WSL2."""

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
from jarvis_voice.piper_training_readiness import piper_training_readiness  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--device-index", type=int, default=0)
    args = parser.parse_args(argv)
    try:
        report = piper_training_readiness(args.work_root.resolve(), device_index=args.device_index)
    except BackendError as exc:
        report = {"ready": False, "code": exc.code}
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
