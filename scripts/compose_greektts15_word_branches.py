#!/usr/bin/env python3
"""Compose disjoint approved GreekTTS MLP and attention branches."""

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
from jarvis_voice.macos.greektts15_training import compose_disjoint_adapter_branches  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--operation-directory", required=True)
    parser.add_argument("--state-directory", required=True)
    parser.add_argument("--output-directory", required=True)
    args = parser.parse_args()
    try:
        report = compose_disjoint_adapter_branches(
            args.model_root,
            args.operation_directory,
            args.state_directory,
            args.output_directory,
        )
    except BackendError as exc:
        report = {"ready": False, "code": exc.code}
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("ready") else 2


if __name__ == "__main__":
    raise SystemExit(main())
