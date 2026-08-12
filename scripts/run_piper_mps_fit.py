#!/usr/bin/env python3
"""Run pinned Piper MPS training with a scoped checkpoint safe-global bridge."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from jarvis_voice.errors import BackendError  # noqa: E402
from jarvis_voice.piper_mps_checkpoint import run_piper_mps_fit  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    try:
        run_piper_mps_fit(arguments)
    except BackendError as exc:
        print(
            json.dumps(
                {
                    "ready": False,
                    "code": exc.code,
                    "checkpoint_loaded": False,
                    "training_started": False,
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
