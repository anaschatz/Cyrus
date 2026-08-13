#!/usr/bin/env python3
"""Prepare the deterministic GreekTTS-1.5/F1 training plan."""

from __future__ import annotations

import argparse
import json

from jarvis_voice.macos.greektts15_training import write_training_plan


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root")
    parser.add_argument("output_path")
    args = parser.parse_args()
    report = write_training_plan(args.dataset_root, args.output_path)
    print(json.dumps({key: report[key] for key in ("schema", "ready", "dataset", "split", "training_policy")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
