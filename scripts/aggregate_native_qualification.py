#!/usr/bin/env python3
"""Aggregate bounded native evidence without weakening qualification gates."""

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
from jarvis_voice.native_evidence_bundle import aggregate_native_evidence  # noqa: E402
from jarvis_voice.qualification import _load_json, write_report  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        runs = [_load_json(path, "qualification.native_bundle_input_invalid") for path in args.input]
        bundle = aggregate_native_evidence(runs)
        write_report(args.output.resolve(), bundle)
    except BackendError as exc:
        result = {
            "ready": False,
            "code": exc.code,
            "source_count": len(args.input),
            "content_captured": False,
        }
        print(json.dumps(result, indent=2, sort_keys=True))
        return 2
    result = {
        "ready": bundle["report"]["ready"],
        "code": bundle["report"]["code"],
        "source_count": bundle["aggregation"]["source_count"],
        "real_voice_turns": bundle["aggregate"]["counters"]["real_voice_turns"],
        "acoustic_samples": bundle["aggregate"]["acoustic_barge_in"]["samples"],
        "maximum_acoustic_ms": bundle["aggregate"]["acoustic_barge_in"]["max_ptt_to_silence_ms"],
        "content_captured": False,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
