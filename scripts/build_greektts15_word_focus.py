#!/usr/bin/env python3
"""Build pinned real-F1 word crops for pronunciation training review."""

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
from jarvis_voice.macos.greektts15_training import (  # noqa: E402
    build_word_focus_bundle,
    build_word_focus_training_plan,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    try:
        report = build_word_focus_bundle(args.plan, args.dataset_root, args.output_root)
        output = Path(args.output_root).expanduser()
        build_word_focus_training_plan(
            Path(args.plan).expanduser().resolve(),
            (output / "word-focus-manifest.json").resolve(),
            (output / "word-focus-training-plan.json").resolve(),
        )
        report["training_plan"] = "word-focus-training-plan.json"
    except BackendError as exc:
        report = {"ready": False, "code": exc.code}
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report.get("ready") else 2


if __name__ == "__main__":
    raise SystemExit(main())
