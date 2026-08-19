"""Keep Finder conflict copies outside the authoritative test collection."""

from __future__ import annotations

from pathlib import Path

_ROOT = Path(__file__).resolve().parent
collect_ignore = [
    str(path)
    for directory in (_ROOT / "tests", _ROOT / "src" / "anamnesis")
    for path in directory.glob("* *.py")
]
