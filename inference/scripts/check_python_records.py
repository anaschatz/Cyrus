#!/usr/bin/env python3
"""Content-safe preflight for installed wheel RECORD integrity."""

from __future__ import annotations

import argparse
import base64
import csv
import importlib.metadata
import json
import os
import re
import stat
from pathlib import Path
from jarvis_inference.python_environment import hash_file


def _canonical(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


def check(root: Path) -> tuple[bool, str | None, str]:
    root = root.absolute()
    site = root / "python/lib/python3.11/site-packages"
    if not root.is_dir() or root.is_symlink() or not site.is_dir():
        return False, None, "record.runtime_layout_invalid"
    distributions = list(importlib.metadata.distributions(path=[str(site)]))
    if not distributions:
        return False, None, "record.distributions_missing"
    for distribution in distributions:
        declared_name = distribution.metadata["Name"]
        if not isinstance(declared_name, str) or not declared_name:
            return False, None, "record.distribution_name_invalid"
        name = _canonical(declared_name)
        record = distribution.read_text("RECORD")
        if record is None:
            return False, name, "record.missing"
        try:
            rows = list(csv.reader(record.splitlines()))
        except csv.Error:
            return False, name, "record.invalid"
        for row in rows:
            if len(row) != 3:
                return False, name, "record.invalid"
            if not row[1]:
                if Path(row[0]).name != "RECORD" or row[2]:
                    return False, name, "record.unhashed_entry"
                continue
            algorithm, separator, encoded = row[1].partition("=")
            if separator != "=" or algorithm != "sha256":
                return False, name, "record.hash_algorithm_invalid"
            installed = Path(os.path.abspath(distribution.locate_file(row[0])))
            try:
                relative = installed.relative_to(root)
            except ValueError:
                return False, name, "record.path_escape"
            current = root
            try:
                for part in relative.parts:
                    current = current / part
                    if stat.S_ISLNK(current.lstat().st_mode):
                        return False, name, "record.symlink"
                if not installed.is_file():
                    return False, name, "record.file_missing"
                size, digest = hash_file(installed)
                expected = base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4)).hex()
                if digest != expected:
                    return False, name, "record.digest_mismatch"
                if row[2] and size != int(row[2]):
                    return False, name, "record.size_mismatch"
            except (OSError, ValueError):
                return False, name, "record.file_unreadable"
    return True, None, "record.all_valid"


def main() -> int:
    parser = argparse.ArgumentParser(prog="check-python-records")
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    ready, package, code = check(args.root)
    print(json.dumps({"ready": ready, "code": code, "package": package}, sort_keys=True, separators=(",", ":")))
    return 0 if ready else 2


if __name__ == "__main__":
    raise SystemExit(main())
