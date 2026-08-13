#!/usr/bin/env python3
"""Train one deterministic 25-step GreekTTS-1.5 pilot branch."""

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
from jarvis_voice.macos.greektts15_training import run_training_pilot  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--gradient-report", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--learning-rate", required=True, type=float, choices=(2.5e-7, 5e-7, 1e-6, 5e-6, 1e-5))
    parser.add_argument(
        "--target-profile",
        choices=("all", "freeze-output-layers", "attention-only"),
        default="all",
    )
    parser.add_argument("--eos-loss-weight", type=int, choices=(1, 16), default=1)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    try:
        report = run_training_pilot(
            args.model_root,
            args.plan,
            args.cache_root,
            args.gradient_report,
            args.output_dir,
            learning_rate=args.learning_rate,
            target_profile=args.target_profile,
            eos_loss_weight=args.eos_loss_weight,
        )
    except BackendError as exc:
        report = {"ready": False, "code": exc.code}
    Path(args.report).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("ready") else 2


if __name__ == "__main__":
    raise SystemExit(main())
