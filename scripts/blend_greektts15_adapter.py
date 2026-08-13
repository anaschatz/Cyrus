#!/usr/bin/env python3
"""Create a bounded GreekTTS adapter interpolation checkpoint."""

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
from jarvis_voice.macos.greektts15_training import blend_adapter_checkpoint  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--candidate-directory", required=True)
    parser.add_argument("--output-directory", required=True)
    parser.add_argument("--candidate-fraction", required=True, type=float, choices=(0.25, 0.5))
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    try:
        report = blend_adapter_checkpoint(
            args.model_root,
            args.candidate_directory,
            args.output_directory,
            candidate_fraction=args.candidate_fraction,
        )
    except BackendError as exc:
        report = {"ready": False, "code": exc.code}
    Path(args.report).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("ready") else 2


if __name__ == "__main__":
    raise SystemExit(main())
