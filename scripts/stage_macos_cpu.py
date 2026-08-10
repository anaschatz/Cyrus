#!/usr/bin/env python3
"""Read-only entry point for sealed macOS CPU prerequisite staging."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from jarvis_voice.config import load_voice_config
from jarvis_voice.errors import BackendError
from jarvis_voice.macos.asr_selection import with_faster_whisper_cpu_smoke
from jarvis_voice.macos.staging import macos_staging_report
from jarvis_voice.qualification import write_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="stage-macos-cpu")
    parser.add_argument("--config", required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--wheelhouse")
    parser.add_argument("--wheelhouse-manifest")
    parser.add_argument("--output")
    return parser


def _emit(value: object, output: str | None) -> None:
    if output:
        write_report(output, value)
    else:
        print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = with_faster_whisper_cpu_smoke(load_voice_config(args.config), args.model_dir)
        report = macos_staging_report(
            config,
            python_executable=args.python,
            wheelhouse=args.wheelhouse,
            wheelhouse_manifest=args.wheelhouse_manifest,
        )
        _emit(report, args.output)
        return 0 if report["ready"] else 2
    except BackendError as exc:
        _emit({"ready": False, "error": exc.to_log_record(), "network_used": False}, args.output)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
