#!/usr/bin/env python3
"""Run deterministic native macOS fixture qualification."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from jarvis_voice.config import load_voice_config
from jarvis_voice.errors import BackendError
from jarvis_voice.macos.asr_selection import with_faster_whisper_cpu_smoke
from jarvis_voice.macos.automated_smoke import run_automated_macos_smoke
from jarvis_voice.qualification import write_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="run-macos-automated-smoke")
    parser.add_argument("--config", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = with_faster_whisper_cpu_smoke(load_voice_config(args.config), args.model_dir)
        report = run_automated_macos_smoke(config, output=args.output)
    except BackendError as exc:
        report = {
            "schema_version": 1,
            "milestone": "5E-A",
            "ready": False,
            "code": exc.code,
            "cases": {},
            "native": {
                "fixture_injected": True,
                "microphone_opened": False,
                "physical_ptt_exercised": False,
                "system_tts_exercised": False,
                "satisfies_interactive_m5e": False,
            },
            "privacy": {
                "temporary_fixture_files_retained": False,
                "raw_audio_persisted": False,
                "transcript_text_persisted": False,
                "tts_text_persisted_in_report": False,
                "protocol_ids_persisted": False,
                "content_captured_in_report": False,
            },
            "network_used": False,
        }
        try:
            write_report(args.output, report)
        except BackendError as output_error:
            report["output_error_code"] = output_error.code
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report.get("ready") else 2


if __name__ == "__main__":
    raise SystemExit(main())
