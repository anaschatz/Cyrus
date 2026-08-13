#!/usr/bin/env python3
"""Build or resume the local GreekTTS-1.5 F1 SNAC token cache."""

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
    TorchSnacEncoder,
    cache_snac_samples,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--snac-root", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--report")
    args = parser.parse_args()
    try:
        encoder = TorchSnacEncoder(args.snac_root)
        report = cache_snac_samples(
            args.plan,
            args.dataset_root,
            args.cache_root,
            encoder,
            limit=args.limit,
        )
    except BackendError as exc:
        report = {"ready": False, "code": exc.code}
    if args.report:
        destination = Path(args.report)
        destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("ready") else 2


if __name__ == "__main__":
    raise SystemExit(main())
