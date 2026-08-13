#!/usr/bin/env python3
"""Run the no-update GreekTTS-1.5 gradient memory probe."""

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
from jarvis_voice.macos.greektts15_training import run_gradient_memory_probe  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    try:
        report = run_gradient_memory_probe(args.model_root, args.plan, args.cache_root)
    except BackendError as exc:
        report = {"ready": False, "code": exc.code}
    destination = Path(args.report)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("ready") else 2


if __name__ == "__main__":
    raise SystemExit(main())
