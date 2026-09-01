"""Make the src-layout package importable without installation."""

import sys
from pathlib import Path

SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

ORCHESTRATOR_SOURCE_ROOT = Path(__file__).resolve().parents[1] / "orchestrator" / "src"
if str(ORCHESTRATOR_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(ORCHESTRATOR_SOURCE_ROOT))

INFERENCE_SOURCE_ROOT = Path(__file__).resolve().parents[1] / "inference" / "src"
if str(INFERENCE_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(INFERENCE_SOURCE_ROOT))
