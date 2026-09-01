#!/usr/bin/env python3
"""Build a content-free private Windows ASR corpus manifest offline."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from jarvis_inference.errors import ApplianceError
from jarvis_inference.windows_asr_corpus import (
    WindowsAsrCorpusError,
    build_corpus_manifest,
    write_corpus_manifest,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        document = build_corpus_manifest(args.corpus_root, args.template)
        write_corpus_manifest(args.output, document)
        print(
            json.dumps(
                {
                    "written": True,
                    "entry_count": len(document["entries"]),
                    "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
                },
                separators=(",", ":"),
            )
        )
        return 0
    except (OSError, ApplianceError, WindowsAsrCorpusError):
        print(json.dumps({"written": False, "code": "audio_corpus_build_failed"}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
