"""Validate and fingerprint an Anamnesis experiment manifest."""

from __future__ import annotations

import argparse
import hashlib
from collections.abc import Sequence
from pathlib import Path

from anamnesis.experiment import ExperimentManifest

DEFAULT_MANIFEST = Path("eval/experiment_manifest.template.json")


def manifest_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate and fingerprint an experiment manifest"
    )
    parser.add_argument("path", nargs="?", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--require-frozen",
        action="store_true",
        help="Reject a valid draft manifest",
    )
    args = parser.parse_args(argv)

    manifest = ExperimentManifest.model_validate_json(
        args.path.read_text(encoding="utf-8")
    )
    if args.require_frozen and manifest.status != "frozen":
        raise ValueError("measured runs require a frozen experiment manifest")

    canonical = manifest.model_dump_json(exclude_none=False)
    digest = hashlib.sha256(canonical.encode()).hexdigest()
    print(
        f"status={manifest.status} phase={manifest.phase} "
        f"scenarios={manifest.scenario_count} sha256={digest}"
    )
    return 0
