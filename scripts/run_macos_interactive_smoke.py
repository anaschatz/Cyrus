#!/usr/bin/env python3
"""Run the guided native macOS CPU PTT smoke and write content-free evidence."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import replace

from jarvis_voice.config import load_voice_config
from jarvis_voice.errors import BackendError
from jarvis_voice.macos.asr_selection import with_faster_whisper_cpu_smoke
from jarvis_voice.macos.interactive_smoke import run_interactive_macos_smoke
from jarvis_voice.qualification import write_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="run-macos-interactive-smoke")
    parser.add_argument("--config", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument("--orchestrator-url")
    parser.add_argument(
        "--trace-stages",
        action="store_true",
        help="Print only allowlisted content-free runtime stage names.",
    )
    parser.add_argument(
        "--ptt-key",
        choices=[*(f"F{index}" for index in range(1, 21)), "CONTROL_OPTION"],
        help="Qualification-only PTT override; the private production config is not modified.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = with_faster_whisper_cpu_smoke(load_voice_config(args.config), args.model_dir)
        if args.orchestrator_url is not None:
            config = replace(config, orchestrator_url=args.orchestrator_url)
        if args.ptt_key is not None:
            config = replace(config, ptt_key=args.ptt_key)
        report = run_interactive_macos_smoke(
            config,
            output=args.output,
            timeout_seconds=args.timeout_seconds,
            stage_sink=(
                (lambda stage: print(f"[smoke.stage] {stage}", flush=True))
                if args.trace_stages
                else None
            ),
        )
    except BackendError as exc:
        report = {
            "schema_version": 1,
            "milestone": "5E",
            "ready": False,
            "code": exc.code,
            "cases": {},
            "privacy": {
                "raw_audio_persisted": False,
                "transcript_text_persisted": False,
                "protocol_ids_persisted": False,
                "content_captured_in_report": False,
            },
            "orchestrator_loopback_only": True,
            "network_used": False,
        }
        write_report(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report.get("ready") else 2


if __name__ == "__main__":
    raise SystemExit(main())
