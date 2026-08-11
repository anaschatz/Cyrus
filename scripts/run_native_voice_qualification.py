#!/usr/bin/env python3
"""Create or evaluate strict native voice qualification evidence."""

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
from jarvis_voice.native_qualification import (  # noqa: E402
    evaluate_native_evidence,
    load_native_evidence,
    write_native_evidence_template,
)
from jarvis_voice.qualification import write_report  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    template = commands.add_parser("template")
    template.add_argument("--platform", choices=("windows", "macos"), required=True)
    template.add_argument("--output", type=Path, required=True)
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--evidence", type=Path, required=True)
    evaluate.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "template":
            write_native_evidence_template(args.output.resolve(), args.platform)
            report = {"ready": False, "code": "qualification.native_template_created", "network_used": False}
        else:
            report = evaluate_native_evidence(load_native_evidence(args.evidence.resolve()))
            if args.output is not None:
                write_report(args.output.resolve(), report)
    except BackendError as exc:
        report = {"ready": False, "code": exc.code, "network_used": False}
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report.get("ready") else 2


if __name__ == "__main__":
    raise SystemExit(main())
