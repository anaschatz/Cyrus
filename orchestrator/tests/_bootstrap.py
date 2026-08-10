from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
for source in (
    ROOT / "orchestrator" / "src",
    ROOT / "src",
    ROOT / "inference" / "src",
    ROOT / "anamnesis" / "src",
):
    value = str(source)
    if value not in sys.path:
        sys.path.insert(0, value)
