#!/usr/bin/env python3
"""Approve the exact reviewed ``λειτουργεί`` crop for bounded training."""

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
    build_approved_singular_operation_training_plan,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-plan", required=True)
    parser.add_argument("--crop-root", required=True)
    parser.add_argument("--combined-plan", required=True)
    parser.add_argument("--cache-plan", required=True)
    args = parser.parse_args()
    try:
        plan = build_approved_singular_operation_training_plan(
            args.base_plan,
            args.crop_root,
            args.combined_plan,
            args.cache_plan,
        )
        report = {
            "ready": True,
            "human_approved": True,
            "training_allowed": True,
            "samples": len(plan["samples"]),
            "audio_sha256": plan["singular_operation_focus"]["audio_sha256"],
        }
    except BackendError as exc:
        report = {"ready": False, "code": exc.code}
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report.get("ready") else 2


if __name__ == "__main__":
    raise SystemExit(main())
