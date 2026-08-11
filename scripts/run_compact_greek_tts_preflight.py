#!/usr/bin/env python3
"""Print the no-install compact Greek TTS preflight as JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from jarvis_voice.errors import BackendError
from jarvis_voice.macos.compact_greek_tts_pilot import compact_greek_tts_preflight


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--storage-root", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = compact_greek_tts_preflight(args.storage_root.resolve())
    except BackendError as exc:
        print(json.dumps({"ready": False, "code": exc.code}, indent=2, sort_keys=True))
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
