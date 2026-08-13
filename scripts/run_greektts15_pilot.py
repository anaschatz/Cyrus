#!/usr/bin/env python3
"""Run the isolated GreekTTS-1.5 MLX qualification pilot."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from jarvis_voice.errors import BackendError  # noqa: E402
from jarvis_voice.macos.greektts15_pilot import (  # noqa: E402
    EXPERIMENTAL_ORPHEUS_SPEAKERS,
    acquire_greektts15_models,
    convert_greektts15_adapter,
    greektts15_pilot_preflight,
    run_greektts15_pilot,
    run_greektts15_pronunciation_qualification,
    run_greektts15_quality_qualification,
)
from jarvis_voice.qualification import write_report  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    preflight = commands.add_parser("preflight")
    preflight.add_argument("--storage-root", required=True)
    preflight.add_argument("--model-root", required=True)
    preflight.add_argument("--base-variant", choices=("4bit", "8bit"), default="4bit")
    preflight.add_argument("--report")
    acquire = commands.add_parser("acquire")
    acquire.add_argument("--storage-root", required=True)
    acquire.add_argument("--model-root", required=True)
    acquire.add_argument("--base-variant", choices=("4bit", "8bit"), default="4bit")
    acquire.add_argument("--report", required=True)
    convert = commands.add_parser("convert-adapter")
    convert.add_argument("--model-root", required=True)
    convert.add_argument("--report", required=True)
    run = commands.add_parser("run")
    run.add_argument("--model-root", required=True)
    run.add_argument("--output-dir", required=True)
    run.add_argument("--base-variant", choices=("4bit", "8bit"), default="4bit")
    run.add_argument("--speaker", choices=EXPERIMENTAL_ORPHEUS_SPEAKERS)
    run.add_argument("--report", required=True)
    run.add_argument("--max-new-tokens", type=int, default=1200)
    run.add_argument("--seed", type=int, default=10001)
    qualify = commands.add_parser("qualify")
    qualify.add_argument("--model-root", required=True)
    qualify.add_argument("--output-dir", required=True)
    qualify.add_argument("--report", required=True)
    qualify.add_argument("--max-new-tokens", type=int, default=1200)
    qualify.add_argument("--seed", type=int, default=10001)
    qualify.add_argument("--adapter-directory")
    qualify.add_argument("--temperature", type=float, choices=(0.0, 0.2, 0.4, 0.6), default=0.6)
    pronunciation = commands.add_parser("pronunciation")
    pronunciation.add_argument("--model-root", required=True)
    pronunciation.add_argument("--dataset-root", required=True)
    pronunciation.add_argument("--output-dir", required=True)
    pronunciation.add_argument("--report", required=True)
    pronunciation.add_argument("--max-new-tokens", type=int, default=1200)
    pronunciation.add_argument("--seed", type=int, default=10001)
    pronunciation.add_argument("--adapter-directory")
    pronunciation.add_argument("--temperature", type=float, choices=(0.0, 0.2, 0.4, 0.6), default=0.4)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "preflight":
            report = greektts15_pilot_preflight(
                args.storage_root,
                args.model_root,
                base_variant=args.base_variant,
            )
        elif args.command == "acquire":
            report = acquire_greektts15_models(
                args.storage_root,
                args.model_root,
                base_variant=args.base_variant,
            )
        elif args.command == "convert-adapter":
            report = convert_greektts15_adapter(args.model_root)
        elif args.command == "qualify":
            report = run_greektts15_quality_qualification(
                args.model_root,
                args.output_dir,
                max_new_tokens=args.max_new_tokens,
                seed=args.seed,
                adapter_directory=args.adapter_directory,
                temperature=args.temperature,
            )
        elif args.command == "pronunciation":
            report = run_greektts15_pronunciation_qualification(
                args.model_root,
                args.dataset_root,
                args.output_dir,
                max_new_tokens=args.max_new_tokens,
                seed=args.seed,
                adapter_directory=args.adapter_directory,
                temperature=args.temperature,
            )
        else:
            report = run_greektts15_pilot(
                args.model_root,
                args.output_dir,
                max_new_tokens=args.max_new_tokens,
                seed=args.seed,
                base_variant=args.base_variant,
                speaker=args.speaker,
            )
        if args.report:
            write_report(args.report, report)
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        return 0 if report["ready"] else 2
    except BackendError as exc:
        print(json.dumps({"ready": False, "code": exc.code}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
