#!/usr/bin/env python3
"""Generate the immutable macOS runtime profile consumed by PyInstaller."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from jarvis_voice.macos.package_profile import build_macos_package_profile


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--helper", type=Path, required=True)
    parser.add_argument("--pyinstaller-version", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    profile = build_macos_package_profile(args.helper, pyinstaller_version=args.pyinstaller_version)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{args.output.name}.", dir=args.output.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as target:
            json.dump(profile, target, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            target.write("\n")
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, args.output)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
