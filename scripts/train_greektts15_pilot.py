#!/usr/bin/env python3
"""Train one deterministic GreekTTS-1.5 pilot branch."""

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
    EOS_LOSS_WEIGHTS,
    EXACT_CORE_STEPS,
    PILOT_LEARNING_RATES,
    PILOT_STEPS,
    TRAINING_SAMPLE_PROFILES,
    TRAINING_TARGET_PROFILES,
    run_training_pilot,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--gradient-report", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--learning-rate", required=True, type=float, choices=sorted(PILOT_LEARNING_RATES))
    parser.add_argument("--steps", type=int, choices=(PILOT_STEPS, EXACT_CORE_STEPS), default=PILOT_STEPS)
    parser.add_argument("--target-profile", choices=sorted(TRAINING_TARGET_PROFILES), default="all")
    parser.add_argument("--eos-loss-weight", type=int, choices=sorted(EOS_LOSS_WEIGHTS), default=1)
    parser.add_argument("--sample-profile", choices=sorted(TRAINING_SAMPLE_PROFILES), default="uniform")
    parser.add_argument("--source-adapter-directory")
    parser.add_argument("--general-stage-index", type=int, default=0)
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
            steps=args.steps,
            target_profile=args.target_profile,
            eos_loss_weight=args.eos_loss_weight,
            sample_profile=args.sample_profile,
            source_adapter_directory=args.source_adapter_directory,
            general_stage_index=args.general_stage_index,
        )
    except BackendError as exc:
        report = {"ready": False, "code": exc.code}
    Path(args.report).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("ready") else 2


if __name__ == "__main__":
    raise SystemExit(main())
