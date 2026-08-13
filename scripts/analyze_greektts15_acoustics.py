#!/usr/bin/env python3
"""Analyze GreekTTS native and synthesized pronunciation fixtures."""

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
from jarvis_voice.macos.greektts15_acoustics import compare_acoustic_sets  # noqa: E402
from jarvis_voice.macos.greektts15_pilot import PRONUNCIATION_REFERENCES  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--blend-25", required=True)
    parser.add_argument("--blend-37-5", required=True)
    parser.add_argument("--candidate")
    parser.add_argument(
        "--candidate-label",
        choices=("lr-2_5e-7", "target-freeze-output-layers"),
        default="lr-2_5e-7",
    )
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    dataset = Path(args.dataset_root).expanduser()
    directories = {
        "baseline": Path(args.baseline).expanduser(),
        "blend-25pct": Path(args.blend_25).expanduser(),
        "blend-37_5pct": Path(args.blend_37_5).expanduser(),
    }
    if args.candidate:
        directories[args.candidate_label] = Path(args.candidate).expanduser()
    files = {label: {} for label in ("native", *directories)}
    for index, reference in enumerate(PRONUNCIATION_REFERENCES, start=1):
        files["native"][reference.fixture_id] = dataset / "audio" / reference.filename
        for label, directory in directories.items():
            files[label][reference.fixture_id] = directory / f"{index:02d}-{reference.fixture_id}.wav"
    try:
        report = compare_acoustic_sets(files)
    except BackendError as exc:
        report = {"ready": False, "code": exc.code}
    Path(args.report).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("ready") else 2


if __name__ == "__main__":
    raise SystemExit(main())
